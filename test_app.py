"""Quick checks. Run:  python test_app.py   (no extra packages needed)"""
import app as m

c = m.app.test_client()
c.post("/login", data={"username":"admin@smartgen.ai", "password":"SmartGen@2026"}, follow_redirects=True)


def post(url, **body):
    r = c.post(url, json=body)
    return r.status_code, r.get_json()


def gen(prompt, purpose="auto"):
    return post("/generate", prompt=prompt, purpose=purpose)


def reset():
    c.post("/stats/reset")


def test_detection():
    assert m.detect_purpose("Butterfly coloring worksheet") == "coloring"
    assert m.detect_purpose("Alphabet flashcard") == "flashcard"
    assert m.detect_purpose("Annual report poster") == "poster"
    assert m.detect_purpose("a dog") == "flashcard"          # unknown -> middle option


def test_generate_smart_is_cheaper():
    reset()
    code, d = gen("Butterfly coloring worksheet")
    assert code == 200 and d["purpose"] == "coloring" and d["source"] == "final"
    assert d["smart_cost"] < d["baseline_cost"] and d["saved_pct"] > 90
    assert d["image"].startswith("data:image/png;base64,")
    assert d["baseline_image"].startswith("data:image/png;base64,")
    assert d["reason"]


def test_estimate_does_not_generate_image():
    reset()
    code, d = post("/estimate", prompt="Butterfly coloring worksheet", purpose="auto")
    assert code == 200 and d["estimate_only"] is True
    assert d["purpose"] == "coloring" and d["smart"]["energy_wh"] < d["maximum"]["energy_wh"]
    assert d["smart"]["cost_inr"] >= 0 and d["equivalent_phone_charge_pct"] >= 0
    assert m.app.test_client().get("/").status_code == 302


def test_resource_estimate_per_generation():
    reset()
    code, d = gen("Fish coloring worksheet")
    for key in ("compute", "energy_wh", "co2_g"):
        assert d["smart"][key] < d["baseline"][key]
    assert d["saved"]["energy_wh"] > 0 and d["saved"]["co2_g"] > 0
    assert abs(m.resources({"size": 1024, "steps": 30})["energy_wh"] - 2.9) < 0.01


def test_poster_has_no_saving():
    reset()
    code, d = gen("Annual report poster")
    assert code == 200 and d["saved_pct"] == 0


def test_override():
    reset()
    code, d = gen("Butterfly coloring worksheet", "poster")
    assert code == 200 and d["purpose"] == "poster" and not d["auto_detected"]


def test_bad_input():
    assert gen("")[0] == 400
    assert gen("x" * 301)[0] == 400
    assert gen("cat", "banana")[0] == 400
    assert post("/finalize", flow_id="nope")[0] == 400
    assert post("/draft", prompt="")[0] == 400


def test_draft_then_final():
    reset()
    code, d1 = post("/draft", prompt="Heart flashcard for kids", purpose="auto")
    assert code == 200 and not d1["done"] and d1["attempts"] == 1
    assert d1["spent_wh"] < d1["final_wh"] / 5          # a draft is far cheaper than the final
    code, d2 = post("/draft", prompt="Heart flashcard for kids", purpose="auto", flow_id=d1["flow_id"])
    assert d2["attempts"] == 2 and d2["image"] != d1["image"]   # a new draft differs
    code, f = post("/finalize", flow_id=d1["flow_id"])
    assert code == 200 and f["source"] == "draft+final" and f["drafts"] == 2
    # baseline = 2 attempts at full poster
    assert abs(f["baseline"]["energy_wh"] - 2 * m.cost(m.PRESETS["poster"])) < 0.01
    assert f["saved_pct"] > 80
    assert post("/finalize", flow_id=d1["flow_id"])[0] == 400   # a flow can only be finalized once


def test_library_reuse():
    reset()
    code, d1 = post("/draft", prompt="Butterfly coloring page", purpose="auto")
    post("/finalize", flow_id=d1["flow_id"])
    code, d2 = post("/draft", prompt="coloring butterfly worksheet for children", purpose="auto")
    assert d2["done"] and d2["source"] == "library"
    assert d2["smart"]["energy_wh"] < 0.01 and d2["saved_pct"] > 99
    code, d3 = gen("butterfly coloring page please")
    assert d3["source"] == "library"
    code, d4 = post("/draft", prompt="Fish coloring page", purpose="auto")
    assert not d4["done"]                                       # a different picture is not reused
    s = c.get("/stats").get_json()
    assert s["reused"] == 2 and s["library_size"] == 1


def test_simulate_calculator_and_stats():
    r = c.post("/simulate", json={"coloring": 6000, "flashcard": 2500, "poster": 500}).get_json()
    assert r["saved_pct"] > 80 and r["saved_kwh"] > 0 and r["saved_co2_kg"] > 0
    assert c.post("/simulate", json={"coloring": "abc"}).get_json()["saved_pct"] == 0
    reset()
    gen("Butterfly coloring worksheet")
    s = c.get("/stats").get_json()
    assert s["requests"] == 1 and s["saved_pct"] > 90


def test_month_simulation():
    code, r = post("/simulate_month", schools=300, per_school=30, avg_attempts=2, reuse_pct=30)
    assert code == 200 and r["days"] == 30 and r["schools"] == 300
    cum = r["cumulative_kwh"]
    for i in range(29):                                          # cumulative lines never go down
        for k in "abcd":
            assert cum[k][i + 1] >= cum[k][i]
    t = r["totals_kwh"]
    assert t["a"] > t["b"] > t["c"] > t["d"]                     # each layer saves more
    b = r["breakdown_kwh"]
    assert abs(sum(b.values()) - r["saved_kwh"]) < 0.05
    assert r["saved_pct"] > 80 and r["phone_charges"] > 0 and len(r["top_schools"]) == 5
    assert sum(len(day) for day in r["events"]) == r["requests"]
    assert m.simulate_month()["totals_kwh"] == m.simulate_month()["totals_kwh"]   # repeatable
    assert post("/simulate_month", schools="abc")[0] == 400
    # more reuse and more attempts both increase the saving
    low = m.simulate_month(reuse_pct=0, avg_attempts=1)["saved_pct"]
    high = m.simulate_month(reuse_pct=60, avg_attempts=3)["saved_pct"]
    assert high > low


def test_every_subject_draws():
    for word in ("butterfly", "flower", "star", "house", "fish", "sun", "tree", "heart", "dog"):
        for purpose in m.PRESETS:
            assert m.generate_image(word, m.PRESETS[purpose], style=purpose).startswith("data:image/png")
        assert m.generate_image(word, m.DRAFT, style="poster", variant=2, tag="DRAFT").startswith("data:image/png")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("All tests passed")
