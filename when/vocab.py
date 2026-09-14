"""通用场景/物体词表，供自动负样本挑选。

这是一次性资产：写一次，各种环境通用，加条目只会更好。
不需要按环境重写——那正是自动挑选要解决的事。
"""

from __future__ import annotations

from typing import List

SCENES: List[str] = [
    "a computer desk with a monitor and keyboard", "a home office workspace",
    "a laptop computer open on a table", "an empty desk surface",
    "a kitchen counter with appliances", "a gas stove with pots and pans",
    "a kitchen sink with dishes", "a refrigerator door", "a microwave oven",
    "a dining table with plates and cutlery", "a cutting board on a counter",
    "kitchen cabinets and drawers", "a countertop with jars and containers",
    "a living room with a sofa", "a coffee table in a living room",
    "a bedroom with a bed and pillows", "a bathroom sink and mirror",
    "a hallway with a closed door", "a staircase", "a balcony railing",
    "a window with daylight coming in", "a plain painted wall",
    "a tiled floor", "a wooden floor", "a carpet",
    "a bookshelf full of books", "a whiteboard with writing",
    "a classroom with desks", "a meeting room table",
    "a supermarket aisle with shelves", "a street with parked cars",
    "a sidewalk with pedestrians", "trees and grass in a park",
    "the inside of a car", "an elevator interior", "a corridor in a building",
]

OBJECTS: List[str] = [
    "a computer keyboard", "a computer mouse on a mousepad", "a computer monitor screen",
    "a laptop computer", "a desk lamp", "a pile of papers and documents",
    "a notebook and a pen", "a smartphone lying flat", "a pair of headphones",
    "a cable lying on a surface", "a power adapter and charger",
    "a cardboard box", "a backpack", "a pair of scissors", "a roll of tape",
    "a potted plant", "a picture frame", "a clock on the wall",
    "a cooking pot", "a frying pan", "a kitchen knife", "a wooden spoon",
    "a plate with food", "a bowl", "a fork and a knife on a table",
    "a bottle of oil or sauce", "a jar of spices", "a loaf of bread",
    "fruits in a bowl", "vegetables on a counter", "a paper towel roll",
    "a dish rack with dishes", "a kettle", "a toaster", "a coffee machine",
    "a trash bin", "a cleaning cloth", "a sponge",
    "a remote control", "a television screen", "a game controller",
    "a stack of folded clothes", "a towel hanging", "a pair of shoes",
]

PEOPLE: List[str] = [
    "a person sitting and working", "a person standing still",
    "a person walking through the room", "a person typing on a keyboard",
    "a person talking", "a person reading",
    "two hands resting on a desk", "a hand reaching for something",
    "an arm passing through the frame", "a person's back facing the camera",
    "an empty room with no people", "nobody is present in the scene",
    "a person cooking at a stove", "a person washing dishes",
    "a person opening a cabinet", "a person carrying something",
    # 笔记本摄像头正对自己的场景
    "a close-up of a person's face", "a person looking at the camera",
    "the head and shoulders of a person indoors", "a person wearing a shirt",
    "a webcam view of someone at a desk", "a plain wall behind a person",
    "a person wearing glasses", "a person with headphones on",
]

QUALITY: List[str] = [
    "a blurry out-of-focus image", "a dark underexposed image",
    "an overexposed bright image", "a close-up of fabric texture",
    "a motion blurred frame", "a plain featureless surface",
    "a shaky camera frame", "a partially obstructed view",
]

BUILTIN_VOCAB: List[str] = SCENES + OBJECTS + PEOPLE + QUALITY


def load_vocab(spec: str = "builtin") -> List[str]:
    """spec 为 'builtin' 用内置词表，否则当成文件路径，一行一条。"""
    if spec == "builtin":
        return list(BUILTIN_VOCAB)
    from pathlib import Path

    lines = Path(spec).read_text(encoding="utf-8").splitlines()
    return [x.strip() for x in lines if x.strip() and not x.startswith("#")]


__all__ = ["BUILTIN_VOCAB", "load_vocab", "SCENES", "OBJECTS", "PEOPLE", "QUALITY"]
