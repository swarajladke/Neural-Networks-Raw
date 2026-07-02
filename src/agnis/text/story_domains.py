"""
Raw AGNIS — src/agnis/text/story_domains.py

Template-based story generators for 4 distinct domains: Animals, Objects, Emotions, and Actions.
"""

import random
from typing import List, Dict, Any

class StoryDomain:
    name: str = "base"
    
    def generate_stories(self, n_stories: int, seed: int) -> List[Dict[str, str]]:
        raise NotImplementedError

    def get_eval_stories(self, n_stories: int, seed: int) -> List[Dict[str, str]]:
        return self.generate_stories(n_stories, seed + 20000)


class AnimalsDomain(StoryDomain):
    name: str = "animals"
    
    def __init__(self):
        self.animals = ["cat", "dog", "bird", "frog", "bear", "owl", "fox"]
        self.names = ["mimi", "toby", "pip", "hops", "bruno", "hoot", "reddy"]
        self.locations = ["park", "garden", "forest", "lake", "house"]
        self.other_animals = ["ant", "bee", "fish", "snail", "mouse"]
        self.actions = ["played", "sang", "danced", "slept", "ran"]

    def generate_stories(self, n_stories: int, seed: int) -> List[Dict[str, str]]:
        rng = random.Random(seed)
        stories = []
        for _ in range(n_stories):
            animal = rng.choice(self.animals)
            name = rng.choice(self.names)
            loc = rng.choice(self.locations)
            other = rng.choice(self.other_animals)
            act = rng.choice(self.actions)
            
            prompt = f"once there was a {animal}"
            target = f" named {name}. {name} liked the {loc}. one day {name} saw a {other}. they {act} together. they were happy.\n"
            stories.append({"prompt": prompt, "target": target})
        return stories


class ObjectsDomain(StoryDomain):
    name: str = "objects"
    
    def __init__(self):
        self.objects = ["toy", "ball", "box", "key", "book", "cup"]
        self.names = ["sam", "ann", "leo", "mia", "ben", "joy"]
        self.locations = ["yard", "room", "box", "bag", "table"]

    def generate_stories(self, n_stories: int, seed: int) -> List[Dict[str, str]]:
        rng = random.Random(seed)
        stories = []
        for _ in range(n_stories):
            obj = rng.choice(self.objects)
            name = rng.choice(self.names)
            loc = rng.choice(self.locations)
            
            prompt = f"here is a {obj}"
            target = f" owned by {name}. {name} lost the {obj} in the {loc}. {name} looked for it. then {name} found it. {name} smiled.\n"
            stories.append({"prompt": prompt, "target": target})
        return stories


class EmotionsDomain(StoryDomain):
    name: str = "emotions"
    
    def __init__(self):
        self.emotions = ["sad", "happy", "scared", "brave"]
        self.locations = ["park", "lake", "woods", "school", "beach"]
        self.objects = ["coin", "shell", "stone", "flower", "leaf"]
        self.other_emotions = ["glad", "proud", "excited", "calm"]

    def generate_stories(self, n_stories: int, seed: int) -> List[Dict[str, str]]:
        rng = random.Random(seed)
        stories = []
        for _ in range(n_stories):
            emo = rng.choice(self.emotions)
            loc = rng.choice(self.locations)
            obj = rng.choice(self.objects)
            other_emo = rng.choice(self.other_emotions)
            
            prompt = f"the child was {emo}"
            target = f" today. the child walked to the {loc}. the child found a {obj}. it made the child feel {other_emo}. the day was good.\n"
            stories.append({"prompt": prompt, "target": target})
        return stories


class ActionsDomain(StoryDomain):
    name: str = "actions"
    
    def __init__(self):
        self.verbs = ["run", "jump", "play", "help", "climb", "swim"]
        self.locations = ["park", "pool", "tree", "hill", "path"]
        self.past_verbs = ["ran", "jumped", "played", "helped", "climbed", "swam"]
        self.objects = ["bench", "tent", "post", "gate", "fence"]

    def generate_stories(self, n_stories: int, seed: int) -> List[Dict[str, str]]:
        rng = random.Random(seed)
        stories = []
        for _ in range(n_stories):
            v = rng.choice(self.verbs)
            loc = rng.choice(self.locations)
            past_v = rng.choice(self.past_verbs)
            obj = rng.choice(self.objects)
            
            prompt = f"the children liked to {v}"
            target = f" in the morning. they went to the {loc}. they {past_v} all day. they rested under the {obj}. it was fun.\n"
            stories.append({"prompt": prompt, "target": target})
        return stories


def get_all_story_domains() -> List[StoryDomain]:
    return [AnimalsDomain(), ObjectsDomain(), EmotionsDomain(), ActionsDomain()]
