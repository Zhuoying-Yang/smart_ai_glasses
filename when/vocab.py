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

    # 容器与饮品 —— 原来只有「调料瓶」，日常最常出现的水瓶杯子全缺
    "a water bottle", "a dark insulated water bottle", "a stainless steel thermos flask",
    "a ceramic mug", "a paper coffee cup", "a drinking glass", "a soda can",
    "a travel tumbler", "a teapot", "a water dispenser",

    # 随身物品
    "a set of keys", "a wallet", "a smartwatch", "a pair of earbuds in a case",
    "an eyeglasses case", "an umbrella", "a face mask", "a lanyard badge",
    "a power bank", "a USB cable",

    # 食物
    "a plate of cooked food", "a sandwich", "a piece of fruit", "a banana",
    "an apple", "a bowl of noodles", "a snack package", "a chocolate bar",
    "a lunch box",

    # 文具
    "a ballpoint pen", "a pencil", "an open notebook", "a hardcover book",
    "a stack of sticky notes", "a paper folder", "a ruler", "a stapler",
    "a highlighter marker",

    # 清洁与卫生
    "a tissue box", "a bottle of hand sanitizer", "a bar of soap",
    "a toothbrush", "a hand towel", "a spray bottle of cleaner",

    # 药品
    "a bottle of pills", "a box of medicine", "a blister pack of tablets",

    # 厨具补充
    "a pair of chopsticks", "a metal spoon", "a measuring cup",
    "a glass jar with a lid", "a kitchen scale",
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
    "a person drinking from a bottle", "a person writing with a pen",
    "a person holding a phone", "a person putting something down on a table",
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

    # 第一人称眼镜视角特有的 —— 鱼眼暗角、手在近处、物体贴脸
    "a fisheye view with dark corners", "a wide angle first person view",
    "a hand close to the camera", "an object held right in front of the camera",
    "a first person view of a desk from above", "a view partly blocked by a large object",
]

BUILTIN_VOCAB: List[str] = SCENES + OBJECTS + PEOPLE + QUALITY

# 条目 -> 类别。给「每类最多选几条」用。
# 句子嵌入的两两相似度都挤在 0.7-0.85，方差太小，MMR 那种基于嵌入的多样性
# 约束推不动排序；按类别限额是更强也更好解释的信号。
CATEGORY: dict[str, str] = {
    **{t: "scene" for t in SCENES},
    **{t: "object" for t in OBJECTS},
    **{t: "people" for t in PEOPLE},
    **{t: "quality" for t in QUALITY},
}


def load_vocab(spec: str = "builtin") -> List[str]:
    """spec 为 'builtin' 用内置词表，否则当成文件路径，一行一条。"""
    if spec == "builtin":
        return list(BUILTIN_VOCAB)
    from pathlib import Path

    lines = Path(spec).read_text(encoding="utf-8").splitlines()
    return [x.strip() for x in lines if x.strip() and not x.startswith("#")]


__all__ = [
    "BUILTIN_VOCAB", "CATEGORY", "load_vocab",
    "SCENES", "OBJECTS", "PEOPLE", "QUALITY",
]
