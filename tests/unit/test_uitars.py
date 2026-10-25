from originblind.agent.uitars import nearest_bid, parse_uitars, resized_to_viewport, smart_resize


def test_smart_resize_multiple_of_28():
    h, w = smart_resize(800, 1280)
    assert h % 28 == 0 and w % 28 == 0
    assert h * w <= 1280 * 800


def test_coord_inverse_roundtrip_scale():
    vp_w, vp_h = 1280, 800
    rh, rw = smart_resize(vp_h, vp_w)
    # a click at the resized-space centre maps back near the viewport centre
    x, y = resized_to_viewport(rw // 2, rh // 2, vp_w, vp_h)
    assert abs(x - vp_w // 2) <= 2 and abs(y - vp_h // 2) <= 2


def test_parse_click_and_type_and_stop():
    assert parse_uitars("Thought: ...\nAction: click(start_box='(100,200)')").kind == "click"
    a = parse_uitars('type(content="hello")')
    assert a.kind == "type" and a.text == "hello"
    assert parse_uitars("finished()").kind == "stop"
    assert parse_uitars("gibberish").kind == "unknown"


def test_nearest_bid_hit_and_miss():
    boxes = [("f1", 0, 0, 10, 10), ("f2", 100, 100, 120, 120)]
    assert nearest_bid(5, 5, boxes) == "f1"
    assert nearest_bid(200, 200, boxes) == "f2"  # nearest centre
