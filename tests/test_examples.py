import importlib
import pytest
import tempfile
import timeit
import time


@pytest.mark.parametrize(
    "example",
    ["egoless", "single_agent", "multi_agent"],
    # TODO: "ego_open_agent" and "human_in_the_loop" are causing aborts, fix later
)
def test_examples(example):
    main = importlib.import_module(f"examples.{example}").main
    main(
        scenarios=["scenarios/loop"],
        sim_name=None,
        headless=True,
        num_episodes=1,
        seed=42,
        max_episode_steps=100,
    )

# pytest -v --full-trace --forked ./tests/test_examples.py::test_frames_per_second

def test_frames_per_second(monkeypatch):
    minimum_fps = 30

    # monkeypatch.setattr(websocket, "WebSocketApp", FakeWebSocketApp)
    main = importlib.import_module("examples.single_agent").main
    cvp = importlib.import_module("examples.single_agent").ChaseViaPointsAgent
   
    scenarios=["scenarios/loop"]
    sim_name=None
    headless=True
    num_episodes=1
    seed=42
    max_episode_steps=10

    start_time = time.perf_counter()
    main(scenarios,sim_name,headless,num_episodes,seed,max_episode_steps)
    elapsed_time = time.perf_counter() - start_time

    time_per_frame = 1 / minimum_fps
    elapsed_time_per_frame = elapsed_time / max_episode_steps / num_episodes
    assert elapsed_time_per_frame < time_per_frame

def test_multi_instance_example():
    main = importlib.import_module("examples.multi_instance").main
    main(
        training_scenarios=["scenarios/loop"],
        evaluation_scenarios=["scenarios/loop"],
        sim_name=None,
        headless=True,
        num_episodes=1,
        seed=42,
    )


def test_rllib_example():
    main = importlib.import_module("examples.rllib").main
    with tempfile.TemporaryDirectory() as result_dir, tempfile.TemporaryDirectory() as model_dir:
        main(
            scenario="scenarios/loop",
            headless=True,
            time_total_s=20,
            rollout_fragment_length=200,
            train_batch_size=200,
            seed=42,
            num_samples=1,
            num_agents=2,
            num_workers=1,
            resume_training=False,
            result_dir=result_dir,
            checkpoint_num=None,
            save_model_path=model_dir,
        )
