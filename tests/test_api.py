from fastapi.testclient import TestClient

from spotter_rate_intelligence.api import app


client = TestClient(app)


def test_health_and_ready():
    assert client.get('/health').json()['status'] == 'ok'
    ready = client.get('/ready')
    assert ready.status_code == 200
    assert ready.json()['status'] == 'ready'


def test_single_prediction_uses_served_core_champion():
    response = client.post('/v1/predict', json={
        'pickup': 'Lexington',
        'delivery': 'Fort Wayne',
        'distance': 360,
        'equipment': 'Dry Van',
        'weight': 32000,
        'date': '2025-12-15',
    })
    assert response.status_code == 200
    body = response.json()
    assert body['predicted_rate'] > 0
    assert body['model_used'] == 'champion_core'
    assert body['status'] in {'SUCCESS', 'RECOVERED_WITH_WARNING'}


def test_invalid_distance_is_rejected_without_server_failure():
    response = client.post('/v1/predict', json={
        'pickup': 'Lexington',
        'delivery': 'Fort Wayne',
        'distance': -1,
        'equipment': 'Dry Van',
        'weight': 32000,
        'date': '2025-12-15',
    })
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'INVALID_INPUT'
    assert body['predicted_rate'] is None
