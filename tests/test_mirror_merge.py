import torch

from backend.mirror_merge import merge_mirror_boxes


def test_mirror_merge_fuses_same_class_and_keeps_other_classes():
    original = torch.tensor([[10., 20., 30., 40., .9, 0.]])
    mirrored = torch.tensor([[69., 20., 89., 40., .6, 0.], [70., 20., 90., 40., .8, 1.]])

    result = merge_mirror_boxes(original, mirrored, width=100)

    expected = torch.tensor([[10.4, 20., 30.4, 40., .75, 0.], [10., 20., 30., 40., .6, 1.]])
    torch.testing.assert_close(result, expected)
    assert mirrored[0, 0] == 69


def test_mirror_merge_accepts_no_boxes():
    empty = torch.empty((0, 6))
    assert merge_mirror_boxes(empty, empty, width=100).shape == (0, 6)
