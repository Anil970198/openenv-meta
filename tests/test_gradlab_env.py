import unittest

from graders import GRADER_REGISTRY
from gradlab_env import GradLabAction, GradLabEnv, TASKS, run_scripted_baseline
from tasks import list_tasks


class GradLabEnvTest(unittest.TestCase):
    def test_all_tasks_reset_and_score_range(self):
        for task_id in TASKS:
            env = GradLabEnv(task_id)
            result = env.reset()
            self.assertFalse(result.done)
            self.assertEqual(result.observation.task_id, task_id)
            self.assertGreaterEqual(env.score(), 0.0)
            self.assertLessEqual(env.score(), 1.0)

    def test_good_overfit_trajectory_scores_high(self):
        env = GradLabEnv("overfit_rescue")
        env.step(GradLabAction(kind="inspect", target="curves", rationale="curves show validation gap evidence"))
        env.step(GradLabAction(kind="inspect", target="config", rationale="config evidence may show missing regularization"))
        env.step(GradLabAction(kind="diagnose", target="root cause", value="overfit and generalization gap"))
        env.step(GradLabAction(kind="repair", target="regularization", value="add dropout and weight decay"))
        env.step(GradLabAction(kind="repair", target="augmentation", value="use stronger augmentation and early stopping"))
        env.step(GradLabAction(kind="evaluate", target="validation", value="run holdout validation learning curve ablation"))
        result = env.step(GradLabAction(kind="finish", value="final plan"))
        self.assertTrue(result.done)
        self.assertGreaterEqual(env.score(), 0.70)

    def test_bad_repeated_action_gets_penalty(self):
        env = GradLabEnv("overfit_rescue")
        first = env.step(GradLabAction(kind="inspect", target="curves"))
        second = env.step(GradLabAction(kind="inspect", target="curves"))
        self.assertGreater(first.reward, second.reward)
        self.assertEqual(second.info["last_action_error"], "repeated action")

    def test_scripted_baseline_is_deterministic_and_bounded(self):
        for task_id in TASKS:
            score_one, rewards_one = run_scripted_baseline(task_id)
            score_two, rewards_two = run_scripted_baseline(task_id)
            self.assertEqual(score_one, score_two)
            self.assertEqual(rewards_one, rewards_two)
            self.assertGreaterEqual(score_one, 0.0)
            self.assertLessEqual(score_one, 1.0)

    def test_explicit_task_catalog_and_graders_exist(self):
        tasks = list_tasks()
        self.assertEqual(len(tasks), 3)
        for task in tasks:
            self.assertIn(task["id"], GRADER_REGISTRY)
            env = GradLabEnv(task["id"])
            grade = GRADER_REGISTRY[task["id"]](env.state())
            self.assertGreaterEqual(grade["score"], 0.0)
            self.assertLessEqual(grade["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
