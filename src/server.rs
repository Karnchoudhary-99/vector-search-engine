//! HTTP REST API for the vector search engine.
//!
//! Run it with `vse serve --port 3000`, then talk to it with curl:
//!
//! ```sh
//! curl -X POST localhost:3000/vectors \
//!      -H 'content-type: application/json' \
//!      -d '{"id": "doc-1", "vector": [0.1, 0.9, 0.0]}'
//! curl -X POST localhost:3000/search/text \
//!      -H 'content-type: application/json' \
//!      -d '{"text": "quick brown fox", "k": 5}'
//! ```
//!
//! Endpoints:
//!
//! ```text
//! GET    /                 web demo (static page)
//! GET    /health           service status
//! GET    /stats            index statistics
//! POST   /vectors          upsert {"id", "vector"}
//! POST   /vectors/text     upsert {"id", "text"} (embedded server-side)
//! DELETE /vectors/:id      soft-delete a vector
//! POST   /search           {"vector", "k"?, "ef"?} -> ranked hits
//! POST   /search/text      {"text", "k"?, "ef"?} -> ranked hits
//! POST   /save             {"path"} persist the index to disk
//! POST   /load             {"path"} replace the index from disk
//! ```

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{Html, Json},
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::{hnsw::HnswIndex, persistence, text, Vector};

/// Shared mutable server state: the index plus the default search beam width.
pub struct ServerState {
    index: RwLock<HnswIndex>,
    default_ef: usize,
}

pub type SharedState = Arc<ServerState>;

impl ServerState {
    /// Wrap an index for serving. `default_ef` is used when a search request
    /// omits `ef`.
    pub fn new(index: HnswIndex, default_ef: usize) -> Self {
        Self {
            index: RwLock::new(index),
            default_ef: default_ef.max(1),
        }
    }

    /// Replace the live index (used by `/load`).
    pub fn replace_index(&self, index: HnswIndex) {
        *self.index.write().expect("index lock poisoned") = index;
    }

    /// Persist the live index to `path` (used on shutdown by `vse serve`).
    pub fn save_to(&self, path: &std::path::Path) -> std::io::Result<()> {
        self.index.read().expect("index lock poisoned").save(path)
    }
}

/// Build the API router for `state`. Used by `vse serve` and by tests.
pub fn router(state: SharedState) -> Router {
    Router::new()
        .route("/", get(demo_page))
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/vectors", post(upsert_vector))
        .route("/vectors/text", post(upsert_text))
        .route("/vectors/:id", delete(delete_vector))
        .route("/search", post(search_vector))
        .route("/search/text", post(search_text))
        .route("/save", post(save_index))
        .route("/load", post(load_index))
        .with_state(state)
}

// ---- response / error types ------------------------------------------------

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

type ApiResult<T> = Result<Json<T>, (StatusCode, Json<ErrorBody>)>;

fn api_error(status: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<ErrorBody>) {
    (status, Json(ErrorBody { error: msg.into() }))
}

fn lock(
    index: &RwLock<HnswIndex>,
) -> Result<std::sync::RwLockReadGuard<'_, HnswIndex>, (StatusCode, Json<ErrorBody>)> {
    index
        .read()
        .map_err(|_| api_error(StatusCode::INTERNAL_SERVER_ERROR, "index lock poisoned"))
}

fn lock_mut(
    index: &RwLock<HnswIndex>,
) -> Result<std::sync::RwLockWriteGuard<'_, HnswIndex>, (StatusCode, Json<ErrorBody>)> {
    index
        .write()
        .map_err(|_| api_error(StatusCode::INTERNAL_SERVER_ERROR, "index lock poisoned"))
}

// ---- handlers --------------------------------------------------------------

async fn demo_page() -> Html<&'static str> {
    Html(include_str!("demo.html"))
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
    service: &'static str,
    version: &'static str,
}

async fn health() -> Json<Health> {
    Json(Health {
        status: "ok",
        service: "vector-search-engine",
        version: env!("CARGO_PKG_VERSION"),
    })
}

#[derive(Serialize)]
struct Stats {
    dim: usize,
    m: usize,
    vectors: usize,
    tombstones: usize,
    max_level: usize,
}

async fn stats(State(state): State<SharedState>) -> ApiResult<Stats> {
    let index = lock(&state.index)?;
    Ok(Json(Stats {
        dim: index.dim(),
        m: index.m(),
        vectors: index.len(),
        tombstones: index.tombstones(),
        max_level: index.max_level(),
    }))
}

#[derive(Deserialize)]
struct UpsertVector {
    id: String,
    vector: Vec<f32>,
}

#[derive(Serialize)]
struct UpsertResponse {
    id: String,
    vectors: usize,
}

async fn upsert_vector(
    State(state): State<SharedState>,
    Json(req): Json<UpsertVector>,
) -> ApiResult<UpsertResponse> {
    if req.id.trim().is_empty() {
        return Err(api_error(StatusCode::BAD_REQUEST, "id must not be empty"));
    }
    let mut index = lock_mut(&state.index)?;
    if req.vector.len() != index.dim() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!(
                "vector has {} dims, index expects {}",
                req.vector.len(),
                index.dim()
            ),
        ));
    }
    index.add(req.id.clone(), Vector::new(req.vector));
    Ok(Json(UpsertResponse {
        id: req.id,
        vectors: index.len(),
    }))
}

#[derive(Deserialize)]
struct UpsertText {
    id: String,
    text: String,
}

async fn upsert_text(
    State(state): State<SharedState>,
    Json(req): Json<UpsertText>,
) -> ApiResult<UpsertResponse> {
    if req.id.trim().is_empty() {
        return Err(api_error(StatusCode::BAD_REQUEST, "id must not be empty"));
    }
    let mut index = lock_mut(&state.index)?;
    let dim = index.dim();
    index.add(req.id.clone(), text::embed(&req.text, dim));
    Ok(Json(UpsertResponse {
        id: req.id,
        vectors: index.len(),
    }))
}

#[derive(Serialize)]
struct DeleteResponse {
    deleted: bool,
    id: String,
}

async fn delete_vector(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> ApiResult<DeleteResponse> {
    let mut index = lock_mut(&state.index)?;
    if index.remove(&id) {
        Ok(Json(DeleteResponse { deleted: true, id }))
    } else {
        Err(api_error(
            StatusCode::NOT_FOUND,
            format!("no live vector with id '{id}'"),
        ))
    }
}

#[derive(Deserialize)]
struct SearchVector {
    vector: Vec<f32>,
    k: Option<usize>,
    ef: Option<usize>,
}

#[derive(Deserialize)]
struct SearchText {
    text: String,
    k: Option<usize>,
    ef: Option<usize>,
}

#[derive(Serialize)]
struct Hit {
    id: String,
    score: f32,
}

#[derive(Serialize)]
struct SearchResponse {
    hits: Vec<Hit>,
}

fn run_search(
    index: &HnswIndex,
    query: &Vector,
    k: Option<usize>,
    ef: Option<usize>,
    default_ef: usize,
) -> Result<Vec<Hit>, String> {
    let k = k.unwrap_or(5);
    if k == 0 {
        return Err("k must be positive".to_string());
    }
    if query.len() != index.dim() {
        return Err(format!(
            "query has {} dims, index expects {}",
            query.len(),
            index.dim()
        ));
    }
    let ef = ef.unwrap_or(default_ef);
    Ok(index
        .search(query, k, ef)
        .into_iter()
        .map(|(id, score)| Hit { id, score })
        .collect())
}

async fn search_vector(
    State(state): State<SharedState>,
    Json(req): Json<SearchVector>,
) -> ApiResult<SearchResponse> {
    let index = lock(&state.index)?;
    let query = Vector::new(req.vector);
    match run_search(&index, &query, req.k, req.ef, state.default_ef) {
        Ok(hits) => Ok(Json(SearchResponse { hits })),
        Err(e) => Err(api_error(StatusCode::BAD_REQUEST, e)),
    }
}

async fn search_text(
    State(state): State<SharedState>,
    Json(req): Json<SearchText>,
) -> ApiResult<SearchResponse> {
    let index = lock(&state.index)?;
    let query = text::embed(&req.text, index.dim());
    match run_search(&index, &query, req.k, req.ef, state.default_ef) {
        Ok(hits) => Ok(Json(SearchResponse { hits })),
        Err(e) => Err(api_error(StatusCode::BAD_REQUEST, e)),
    }
}

#[derive(Deserialize)]
struct PathRequest {
    path: String,
}

#[derive(Serialize)]
struct PathResponse {
    path: String,
    vectors: usize,
}

async fn save_index(
    State(state): State<SharedState>,
    Json(req): Json<PathRequest>,
) -> ApiResult<PathResponse> {
    if req.path.trim().is_empty() {
        return Err(api_error(StatusCode::BAD_REQUEST, "path must not be empty"));
    }
    let index = lock(&state.index)?;
    let path = PathBuf::from(&req.path);
    index.save(&path).map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("save failed: {e}"),
        )
    })?;
    Ok(Json(PathResponse {
        path: req.path,
        vectors: index.len(),
    }))
}

async fn load_index(
    State(state): State<SharedState>,
    Json(req): Json<PathRequest>,
) -> ApiResult<PathResponse> {
    if req.path.trim().is_empty() {
        return Err(api_error(StatusCode::BAD_REQUEST, "path must not be empty"));
    }
    let path = PathBuf::from(&req.path);
    let loaded = persistence::load(&path)
        .map_err(|e| api_error(StatusCode::BAD_REQUEST, format!("load failed: {e}")))?;
    let index = match loaded {
        persistence::LoadedIndex::Hnsw(idx) => idx,
        persistence::LoadedIndex::Brute(_) => {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "file holds a brute-force index, expected HNSW",
            ));
        }
    };
    let vectors = index.len();
    state.replace_index(index);
    Ok(Json(PathResponse {
        path: req.path,
        vectors,
    }))
}
