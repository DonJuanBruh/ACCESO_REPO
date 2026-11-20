import unittest

import ai_assistant
import scenarios


class GuidedAssistantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = scenarios.prepare_base()
        cls.ctx = ai_assistant.Context(cls.base, scenarios.compute_scenario(cls.base, scenarios.parameters()))

    def ask(self, question):
        return ai_assistant.respond(question, [], self.ctx, force_offline=True)

    def test_answers_cite_the_scenario(self):
        reply = self.ask("Resume el escenario actual")
        self.assertIn(self.ctx.id, reply["text"])
        self.assertEqual(reply["source"], "Modo guiado")

    def test_answers_contain_no_unverified_figures(self):
        for question in ai_assistant.guided_questions(self.ctx):
            reply = self.ask(question)
            self.assertEqual(reply["verification"]["unverified"], [], question)

    def test_center_explanation_uses_engine_tool(self):
        reply = self.ask("¿Por qué el centro de salud número 1 quedó ahí?")
        self.assertIn("explain_center", [t["tool"] for t in reply["tools"]])

    def test_road_closure_question_simulates_closure(self):
        reply = self.ask("¿Qué pasa si se cae el puente?")
        self.assertIn("simulate_road_closure", [t["tool"] for t in reply["tools"]])

    def test_every_declared_tool_has_an_implementation(self):
        declared = {t["function"]["name"] for t in ai_assistant.TOOLS}
        self.assertEqual(declared, set(ai_assistant.FUNCTIONS))


if __name__ == "__main__":
    unittest.main()
