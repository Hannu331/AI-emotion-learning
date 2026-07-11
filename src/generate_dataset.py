"""
Generates a synthetic labeled dataset of student statements for each
of the 5 target emotions, using local sentence templates + slot-filling
(no API calls, no quota limits).

v3: expanded templates per emotion (24 each, mixing clean grammar with
casual/internet-style phrasing: contractions without apostrophes, "ugh",
"omg", sentence fragments, etc.) so the trained model generalizes to
real freeform student text instead of memorizing a narrow set of
sentence structures.

Run from project root:
    python -m src.generate_dataset
"""

import os
import csv
import random

random.seed(42)

OUTPUT_PATH = "data/dataset.csv"
EXAMPLES_PER_EMOTION = 220

SUBJECTS = [
    "recursion", "this calculus problem", "the essay outline", "photosynthesis",
    "this Python error", "the group project plan", "linear algebra", "this lab report",
    "the history reading", "SQL joins", "this grammar rule", "thermodynamics",
    "the coding assignment", "statistics homework", "this chemistry equation",
    "the research paper thesis", "object-oriented programming", "this proof",
    "the presentation slides", "machine learning basics", "this French verb tense",
    "the biology diagram", "algorithms and complexity", "this economics concept",
    "the exam material", "data structures", "this physics problem set",
    "this loop", "the assignment instructions", "binary search", "this API",
    "the lecture notes", "my code", "this formula", "the reading assignment",
]

CONTEXT_TAGS = [
    "for my assignment", "before the exam", "in today's lecture", "for the group project",
    "while revising my draft", "during lab", "for homework", "in class today",
    "before the deadline", "while studying alone", "", "", "right now", "again",
]

TEMPLATES = {
    "Bored": [
        "I've been staring at {subject} {context} and I just can't focus anymore.",
        "Honestly {subject} feels so repetitive, I keep zoning out.",
        "Not gonna lie, {subject} is putting me to sleep {context}.",
        "I don't feel motivated to keep going through {subject} right now.",
        "This is dragging on forever, {subject} just isn't interesting to me.",
        "I keep checking the clock instead of working on {subject}.",
        "{subject} feels like it's going nowhere, I've lost interest.",
        "Meh, {subject} again... I've done this a hundred times.",
        "I can't bring myself to care about {subject} today.",
        "Everything about {subject} feels flat and uninteresting right now.",
        "I'm just going through the motions with {subject}, not really engaged.",
        "Same old stuff with {subject}, nothing new is grabbing my attention.",
        "ugh {subject} again, so boring",
        "meh another {subject}, we already know this stuff",
        "cant focus on {subject} at all rn, so dull",
        "yawn... {subject} is not holding my attention today",
        "smh {subject} is dragging on forever",
        "not even a little interested in {subject} tbh",
        "zoned out like 3 times reading about {subject}",
        "{subject} is just... nothing new, kinda bored",
        "why is {subject} so dry, i can barely stay awake",
        "skimming through {subject} bc its so uninteresting",
        "this {subject} lecture is a snoozefest",
        "i keep drifting off thinking about literally anything else during {subject}",
    ],
    "Confident": [
        "I think I've finally got the hang of {subject}!",
        "Pretty sure I nailed {subject} {context}, feeling good about it.",
        "I actually understand {subject} now, this makes sense.",
        "I'm ready to tackle {subject} head on, no worries.",
        "Feels great knowing I can handle {subject} on my own now.",
        "I walked through {subject} step by step and it clicked.",
        "I'm confident I can explain {subject} to someone else now.",
        "This part of {subject} used to trip me up, but not anymore.",
        "I double-checked my work on {subject} and it all adds up.",
        "I feel solid going into the next part of {subject}.",
        "Finally cracked {subject}, I know exactly what I'm doing.",
        "I've got a good grip on {subject} {context}.",
        "omg i think i finally figured out {subject}!!",
        "yesss got {subject} working, feeling good",
        "pretty sure i can handle {subject} no problem now",
        "{subject} finally makes sense, i got this",
        "nailed it, {subject} works exactly how it should now",
        "im actually feeling good about {subject} for once",
        "took a while but i totally get {subject} now",
        "no more stress over {subject}, i know what im doing",
        "just aced a practice question on {subject}, feeling great",
        "{subject} clicked for me and now im flying through it",
        "solid on {subject} now, ready for whatevers next",
        "im not even worried about {subject} anymore, got it down",
    ],
    "Confused": [
        "I don't understand {subject} at all {context}.",
        "Wait, I'm lost — how does {subject} even work?",
        "None of this makes sense, especially {subject}.",
        "I've reread {subject} three times and I'm still confused.",
        "Can someone explain {subject}? I have no idea what's going on.",
        "I thought I understood {subject}, but now I'm totally stuck.",
        "{subject} isn't clicking for me, what am I missing?",
        "I keep getting mixed up on {subject}, it's unclear.",
        "This part of {subject} just doesn't add up to me.",
        "I'm not sure what the question is even asking about {subject}.",
        "My brain just blanks every time I look at {subject}.",
        "Something about {subject} isn't making sense and I can't pinpoint why.",
        "wait what?? {subject} makes zero sense to me",
        "im so lost with {subject} rn, nothing is clicking",
        "huh, i dont get {subject} at all",
        "{subject} is just not making sense no matter how many times i read it",
        "someone explain {subject} pls im so confused",
        "brain totally blanking on {subject}, whats going on",
        "idk what im even looking at with {subject}",
        "this doesnt add up, {subject} seems contradictory",
        "im mixing up {subject} with something else i think, so unclear",
        "cant tell whats going on with {subject} at all",
        "{subject} is a total blur to me right now",
        "why does {subject} feel like a different language",
    ],
    "Curious": [
        "Wait, that's actually interesting — how does {subject} really work?",
        "I want to dig deeper into {subject}, there's more to it I think.",
        "This made me wonder about {subject}, can you explain more?",
        "I'm intrigued by {subject}, is there a real-world example?",
        "Now I'm curious how {subject} connects to what we learned before.",
        "What happens if we tweak {subject} a little, would it change things?",
        "I'd love to explore {subject} further, it's more fascinating than I expected.",
        "Is there a reason {subject} works this way? I want to understand the why.",
        "This part of {subject} got me thinking, what else is related to it?",
        "I keep wanting to ask more questions about {subject}.",
        "Something about {subject} sparked my interest, tell me more.",
        "I wonder how {subject} would apply outside of class.",
        "ooh wait thats interesting, how does {subject} actually work under the hood?",
        "kinda curious now, is there more to {subject} than what we covered?",
        "hmm what if we changed something in {subject}, would that break it?",
        "this got me thinking about {subject}, wanna know more",
        "is there a cool real world use case for {subject}?",
        "why does {subject} even work like that, genuinely curious",
        "wait so how is {subject} related to the other stuff we learned",
        "now i wanna go down a rabbit hole on {subject}",
        "that part about {subject} was surprisingly interesting, more please",
        "so is {subject} used in real projects or just theory",
        "what would happen if we pushed {subject} further, curious to see",
        "never thought about {subject} like that, want to learn more",
    ],
    "Frustrated": [
        "I've tried everything and {subject} still isn't working {context}.",
        "This is so annoying, {subject} keeps breaking no matter what I do.",
        "I'm stuck on {subject} and it's making me want to give up.",
        "Nothing I try fixes {subject}, I'm getting really frustrated.",
        "I've spent hours on {subject} and I'm no closer to solving it.",
        "Why does {subject} keep going wrong every single time?",
        "I'm so close to giving up on {subject}, it's exhausting.",
        "This error in {subject} is driving me crazy.",
        "I keep hitting the same wall with {subject} and I'm over it.",
        "I redo {subject} again and again and it still doesn't work.",
        "I'm out of patience with {subject} at this point.",
        "{subject} just isn't cooperating no matter what I try {context}.",
        "ugh why wont {subject} just work, ive tried everything",
        "so done with {subject}, nothing is working no matter what",
        "this is so annoying, {subject} keeps breaking on me",
        "literally about to give up on {subject}, im exhausted",
        "why does {subject} keep failing every single time i try",
        "ive redone {subject} like 5 times and still broken",
        "so frustrated rn, {subject} just wont cooperate",
        "im losing my mind over {subject}, nothing helps",
        "keep hitting the same error with {subject} ugh",
        "at my limit with {subject}, ive tried literally everything",
        "{subject} is being so difficult for no reason",
        "why wont {subject} just work already, im so over it",
    ],
}


def generate_for_emotion(emotion: str, n: int):
    templates = TEMPLATES[emotion]
    combos = set()
    results = []
    attempts = 0
    max_attempts = n * 30

    while len(results) < n and attempts < max_attempts:
        attempts += 1
        template = random.choice(templates)
        subject = random.choice(SUBJECTS)
        context = random.choice(CONTEXT_TAGS)
        text = template.format(subject=subject, context=context)
        text = " ".join(text.split())  # collapse double spaces from empty context
        text = text.replace(" .", ".")

        key = (template, subject, context)
        if key not in combos:
            combos.add(key)
            results.append(text)

    return results


def main():
    os.makedirs("data", exist_ok=True)
    rows = []

    for emotion in TEMPLATES:
        examples = generate_for_emotion(emotion, EXAMPLES_PER_EMOTION)
        for text in examples:
            rows.append({"text": text, "emotion": emotion})
        print(f"{emotion}: generated {len(examples)} examples")

    random.shuffle(rows)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "emotion"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Saved {len(rows)} examples to {OUTPUT_PATH}")
    from collections import Counter
    print("Class distribution:", Counter(r["emotion"] for r in rows))


if __name__ == "__main__":
    main()