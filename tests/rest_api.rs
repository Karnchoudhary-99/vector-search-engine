//! Integration tests for the REST API: upsert, search (vector + text),
//! delete, stats, and save/load — all through the real axum router.

use axum::body::{to_bytes, Body};
use http::{Request, StatusCode};
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceExt;
use vector_search_engine::{
    hnsw::HnswIndex,
    server::{router, ServerState, SharedState},
};

fn state() -> SharedState {
    // 4-D toy index with a fixed seed for deterministic routing.
    Arc::new(ServerState::new(HnswIndex::with_seed(4, 8, 50, 42), 20))
}

async fn post(state: &SharedState, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = router(state.clone()).oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    (status, json)
}

async fn get(state: &SharedState, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
    let resp = router(state.clone()).oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    // The demo page is HTML, not JSON.
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

async fn delete(state: &SharedState, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("DELETE")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = router(state.clone()).oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn health_and_stats() {
    let s = state();
    let (status, body) = get(&s, "/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");

    let (status, body) = get(&s, "/stats").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["vectors"], 0);
    assert_eq!(body["dim"], 4);
}

#[tokio::test]
async fn demo_page_is_served() {
    let s = state();
    let req = Request::builder().uri("/").body(Body::empty()).unwrap();
    let resp = router(s).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let html = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(html.contains("Vector Search Engine"));
}

#[tokio::test]
async fn upsert_then_vector_search() {
    let s = state();
    for (id, v) in [
        ("a", vec![1.0, 0.0, 0.0, 0.0]),
        ("b", vec![0.0, 1.0, 0.0, 0.0]),
    ] {
        let (status, body) = post(&s, "/vectors", json!({"id": id, "vector": v})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["id"], id);
    }

    let (status, body) = post(
        &s,
        "/search",
        json!({"vector": [0.95, 0.05, 0.0, 0.0], "k": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let hits = body["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0]["id"], "a");
    assert!(hits[0]["score"].as_f64().unwrap() > hits[1]["score"].as_f64().unwrap());
}

#[tokio::test]
async fn upsert_text_then_text_search() {
    let s = state();
    let (status, _) = post(
        &s,
        "/vectors/text",
        json!({"id": "doc-1", "text": "the quick brown fox jumps"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    post(
        &s,
        "/vectors/text",
        json!({"id": "doc-2", "text": "quantum field theory notes"}),
    )
    .await;

    let (status, body) = post(
        &s,
        "/search/text",
        json!({"text": "quick brown fox", "k": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let hits = body["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0]["id"], "doc-1");
}

#[tokio::test]
async fn upsert_is_idempotent_for_same_id() {
    let s = state();
    post(
        &s,
        "/vectors",
        json!({"id": "x", "vector": [1.0, 0.0, 0.0, 0.0]}),
    )
    .await;
    let (status, body) = post(
        &s,
        "/vectors",
        json!({"id": "x", "vector": [0.0, 1.0, 0.0, 0.0]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["vectors"], 1); // still one live vector

    let (_, stats) = get(&s, "/stats").await;
    assert_eq!(stats["tombstones"], 1);

    let (status, body) = post(
        &s,
        "/search",
        json!({"vector": [0.0, 1.0, 0.0, 0.0], "k": 1}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["hits"][0]["id"], "x");
    assert!(body["hits"][0]["score"].as_f64().unwrap() > 0.99);
}

#[tokio::test]
async fn delete_hides_vector() {
    let s = state();
    post(
        &s,
        "/vectors",
        json!({"id": "gone", "vector": [1.0, 0.0, 0.0, 0.0]}),
    )
    .await;
    post(
        &s,
        "/vectors",
        json!({"id": "stay", "vector": [0.0, 1.0, 0.0, 0.0]}),
    )
    .await;

    let (status, body) = delete(&s, "/vectors/gone").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["deleted"], true);

    let (status, _) = delete(&s, "/vectors/gone").await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, body) = post(
        &s,
        "/search",
        json!({"vector": [1.0, 0.0, 0.0, 0.0], "k": 5}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let hits = body["hits"].as_array().unwrap();
    assert!(hits.iter().all(|h| h["id"] != "gone"));
}

#[tokio::test]
async fn wrong_dimensions_are_rejected() {
    let s = state();
    let (status, body) = post(&s, "/vectors", json!({"id": "bad", "vector": [1.0, 2.0]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].as_str().unwrap().contains("dims"));

    let (status, _) = post(&s, "/search", json!({"vector": [1.0]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, _) = post(
        &s,
        "/vectors",
        json!({"id": "  ", "vector": [1.0, 0.0, 0.0, 0.0]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn save_and_load_roundtrip() {
    let s = state();
    post(
        &s,
        "/vectors",
        json!({"id": "p1", "vector": [1.0, 0.0, 0.0, 0.0]}),
    )
    .await;
    let p = std::env::temp_dir().join("vse-api-test.vse");

    let (status, body) = post(&s, "/save", json!({"path": p.to_str().unwrap()})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["vectors"], 1);

    // Wipe the live index via a fresh state, then load the file into it.
    let s2 = state();
    let (status, body) = post(&s2, "/load", json!({"path": p.to_str().unwrap()})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["vectors"], 1);

    let (status, body) = post(
        &s2,
        "/search",
        json!({"vector": [0.9, 0.1, 0.0, 0.0], "k": 1}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["hits"][0]["id"], "p1");
    std::fs::remove_file(p).ok();
}

#[tokio::test]
async fn loading_missing_file_fails() {
    let s = state();
    let (status, body) = post(&s, "/load", json!({"path": "/does/not/exist.vse"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].as_str().unwrap().contains("load failed"));
}
