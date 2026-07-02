"""
Raw AGNIS — src/agnis/text/text_domains.py

Deterministic text domain generators for Prose, Code, Arithmetic, and Dialogue.
"""

import random
from typing import List

class TextDomain:
    name: str = "base"

    def generate(self, n_chars: int, seed: int) -> str:
        raise NotImplementedError

    def get_eval_text(self, n_chars: int, seed: int) -> str:
        # Use a distinct offset seed for evaluation to avoid overlap/leakage
        return self.generate(n_chars, seed + 10000)


class ProseDomain(TextDomain):
    name: str = "prose"

    def __init__(self):
        self.subjects = ['the cat', 'the dog', 'a bird', 'the fish', 'a fox', 'the bear', 'a frog', 'the owl', 'the ant', 'a bee']
        self.verbs = ['sat on', 'ran to', 'saw', 'found', 'liked', 'ate', 'chased', 'watched', 'heard', 'felt']
        self.objects = ['the mat', 'the park', 'a tree', 'the hill', 'the lake', 'a rock', 'the cave', 'the nest', 'a leaf', 'the sun']

    def generate(self, n_chars: int, seed: int) -> str:
        rng = random.Random(seed)
        text_parts = []
        current_len = 0
        
        while current_len < n_chars:
            subj = rng.choice(self.subjects)
            verb = rng.choice(self.verbs)
            obj = rng.choice(self.objects)
            
            # Sentence formats
            ending = rng.choice(['. ', '.\n'])
            sentence = f"{subj} {verb} {obj}{ending}"
            text_parts.append(sentence)
            current_len += len(sentence)
            
        full_text = "".join(text_parts)
        return full_text[:n_chars]


class CodeDomain(TextDomain):
    name: str = "code"

    def __init__(self):
        self.functions = ['add', 'sub', 'mul', 'div', 'inc', 'dec', 'neg', 'abs', 'sqr', 'dbl']
        self.vars = ['x', 'y', 'z', 'a', 'b', 'n', 'm', 'k', 'p', 'q']
        self.ops = ['+', '-', '*']
        self.vals = ['0', '1', '2', '3', '4', '5']

    def generate(self, n_chars: int, seed: int) -> str:
        rng = random.Random(seed)
        text_parts = []
        current_len = 0
        
        while current_len < n_chars:
            block_type = rng.choice(['def', 'if', 'assign'])
            if block_type == 'def':
                func = rng.choice(self.functions)
                v1 = rng.choice(self.vars)
                v2 = rng.choice(self.vars)
                while v1 == v2:
                    v2 = rng.choice(self.vars)
                op = rng.choice(self.ops)
                code_block = f"def {func}({v1}, {v2}):\n  return {v1} {op} {v2}\n"
            elif block_type == 'if':
                v = rng.choice(self.vars)
                val = rng.choice(self.vals)
                v_target = rng.choice(self.vars)
                code_block = f"if {v} = {val}:\n  {v_target} = {val}\n"
            else:
                v = rng.choice(self.vars)
                val = rng.choice(self.vals)
                op = rng.choice(self.ops)
                v2 = rng.choice(self.vars)
                code_block = f"{v} = {v2} {op} {val}\n"
                
            text_parts.append(code_block)
            current_len += len(code_block)
            
        full_text = "".join(text_parts)
        return full_text[:n_chars]


class ArithmeticDomain(TextDomain):
    name: str = "arithmetic"

    def __init__(self):
        self.ops = ['+', '-', '*']

    def generate(self, n_chars: int, seed: int) -> str:
        rng = random.Random(seed)
        text_parts = []
        current_len = 0
        
        while current_len < n_chars:
            op = rng.choice(self.ops)
            if op == '+':
                a = rng.randint(1, 99)
                b = rng.randint(1, 99)
                ans = a + b
            elif op == '-':
                a = rng.randint(50, 99)
                b = rng.randint(1, 49)
                ans = a - b
            else:
                a = rng.randint(1, 12)
                b = rng.randint(1, 8)
                ans = a * b
                
            expr = f"{a} {op} {b} = {ans}\n"
            text_parts.append(expr)
            current_len += len(expr)
            
        full_text = "".join(text_parts)
        return full_text[:n_chars]


class DialogueDomain(TextDomain):
    name: str = "dialogue"

    def __init__(self):
        self.qa_pairs = [
            ("what is your name", "my name is alice"),
            ("where do you live", "i live in the city"),
            ("how are you", "i am doing well"),
            ("what do you do", "i write code"),
            ("do you like cats", "yes i love cats"),
            ("can you see the sun", "yes it is very bright"),
            ("is it warm outside", "no it is cold"),
            ("will it rain today", "maybe it looks cloudy"),
            ("who is there", "it is only me"),
            ("what time is it", "it is noon")
        ]

    def generate(self, n_chars: int, seed: int) -> str:
        rng = random.Random(seed)
        text_parts = []
        current_len = 0
        
        while current_len < n_chars:
            q, a = rng.choice(self.qa_pairs)
            turn = f"q: {q}?\na: {a}.\n"
            text_parts.append(turn)
            current_len += len(turn)
            
        full_text = "".join(text_parts)
        return full_text[:n_chars]


def get_all_domains() -> List[TextDomain]:
    return [ProseDomain(), CodeDomain(), ArithmeticDomain(), DialogueDomain()]
