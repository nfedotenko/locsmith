import json, sys, unittest, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import locsmith as L


class Validate(unittest.TestCase):
    def test_placeholders_and_tags(self):
        it = L.Item("k", "<b>{name}</b> has %d gold\\n[color=red]ok[/color] {{x}}")
        good = "<b>{name}</b> имеет %d золота\\n[color=red]ок[/color] {{x}}"
        self.assertEqual(L.validate(it, good, []), [])
        bad = "<b>{имя}</b> имеет %s золота"
        issues = " ".join(L.validate(it, bad, []))
        self.assertIn("lost placeholders", issues)
        self.assertIn("{name}", issues)
        self.assertIn("unexpected", issues)

    def test_reorder_allowed(self):
        it = L.Item("k", "You picked up %d x %s")
        self.assertEqual(L.validate(it, "Подобрано: %s x %d", []), [])

    def test_length_and_untranslated(self):
        it = L.Item("k", "Settings", max_len=5)
        self.assertTrue(any("too long" in i for i in L.validate(it, "Настройки", [])))
        self.assertTrue(any("untranslated" in i for i in L.validate(L.Item("k", "Settings"), "Settings", [])))

    def test_glossary_inflection_and_strict(self):
        g = [L.GlossaryEntry("Aether Shard", "Осколок Эфира"), L.GlossaryEntry("Warden", "Страж", strict=True)]
        it = L.Item("k", "You need 5 more Aether Shards, Warden.")
        self.assertEqual(L.validate(it, "Нужно ещё 5 осколков Эфира, Страж.", g), [])
        issues = L.validate(it, "Нужно ещё 5 кристаллов, Хранитель.", g)
        self.assertEqual(len(issues), 2)

    def test_missing_and_newlines(self):
        self.assertEqual(L.validate(L.Item("k", "a"), None, []), ["missing in model output"])
        it = L.Item("k", "line one\nline two")
        self.assertIn("line break count changed", L.validate(it, "строка один строка два", []))


class Parse(unittest.TestCase):
    def test_fenced_and_chatty(self):
        self.assertEqual(L.parse_json_object('```json\n{"a": "б"}\n```'), {"a": "б"})
        self.assertEqual(L.parse_json_object('Sure! {"a": "б"} hope it helps'), {"a": "б"})
        self.assertEqual(L.parse_json_object("no json here"), {})


class ModelIO(unittest.TestCase):
    def _capture(self, **kw):
        sent = {}

        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps(
                {"choices": [{"message": {"content": '{"k": "v"}'}}]}).encode()

        def fake_urlopen(req, timeout=0):
            sent["url"] = req.full_url
            sent["body"] = json.loads(req.data)
            sent["auth"] = req.headers.get("Authorization")
            return Resp()

        orig = L.urllib.request.urlopen
        L.urllib.request.urlopen = fake_urlopen
        try:
            llm = L.make_llm("http://host/v1", "gemma4:12b", None, 0.2, 60, **kw)
            self.assertEqual(L.parse_json_object(llm([{"role": "user", "content": "hi"}])), {"k": "v"})
        finally:
            L.urllib.request.urlopen = orig
        return sent

    def test_reasoning_effort_sent_when_set(self):
        sent = self._capture(reasoning="none")
        self.assertEqual(sent["url"], "http://host/v1/chat/completions")
        self.assertEqual(sent["body"]["reasoning_effort"], "none")

    def test_reasoning_effort_omitted_by_default(self):
        self.assertNotIn("reasoning_effort", self._capture()["body"])


class Pipeline(unittest.TestCase):
    def test_retry_fixes_broken_placeholder(self):
        calls = []

        def fake_llm(messages):
            payload = json.loads(messages[-1]["content"])
            calls.append(payload)
            if "problems_to_fix" in payload:
                return '{"shop.buy": "Купить за {price} осколков Эфира"}'
            return '{"shop.buy": "Купить за {цена} осколков Эфира", "ui.play": "Играть"}'

        items = [L.Item("shop.buy", "Buy for {price} Aether Shards"), L.Item("ui.play", "Play", max_len=12)]
        g = [L.GlossaryEntry("Aether Shard", "Осколок Эфира")]
        res = L.translate(items, g, fake_llm, src="English", tgt="Russian", style="", batch_size=10,
                          retries=2, cache={}, log=lambda m: None)
        self.assertEqual([r.status for r in res], ["ok", "ok"])
        self.assertEqual(res[0].attempts, 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["glossary"], [{"source": "Aether Shard", "target": "Осколок Эфира"}])

    def test_cache_skips_unchanged(self):
        def boom(messages):
            raise AssertionError("model must not be called")
        items = [L.Item("ui.play", "Play")]
        cache = {"ui.play": {"key": "ui.play", "source": "Play", "target": "Играть", "status": "ok"}}
        res = L.translate(items, [], boom, src="English", tgt="Russian", style="", batch_size=10,
                          retries=1, cache=cache, log=lambda m: None)
        self.assertTrue(res[0].cached)
        self.assertEqual(res[0].target, "Играть")

    def test_gives_up_after_retries(self):
        res = L.translate([L.Item("k", "Hi {name}")], [], lambda m: '{"k": "Привет"}', src="English",
                          tgt="Russian", style="", batch_size=10, retries=2, cache={}, log=lambda m: None)
        self.assertEqual(res[0].status, "review")
        self.assertEqual(res[0].attempts, 3)


if __name__ == "__main__":
    unittest.main()
