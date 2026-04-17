# AI Digest — April 2026
### Speaker Script

---

## Slide 1 — Introduction

Good morning / Good afternoon everyone. Today I want to walk you through a digest of the most significant developments in AI this month. April 2026 has been packed: major model releases, unexpected safety incidents, mathematical breakthroughs, and new players entering the frontier race. Let's get into it.

---

## Slide 2 — Agenda

We'll cover nine topics today. We'll start with Anthropic and what the Opus 4.7 release notes didn't tell you. Then we'll look at mathematics — GPT-5.4 solved a problem in 80 minutes that took a human seven years. We'll cover security, new models from Meta and Google, the Mythos containment incident, Musk's plans, and a new scientific tool from OpenAI.

---

## Slide 3 — Anthropic Opus 4.7

Anthropic released Opus 4.7 with the usual messaging: better code, better vision, same pricing. But if you read closely, there are several important details buried in the release.

First, **a new tokenizer**. The same text now generates 1.0 to 1.35 times more tokens. Code and non-English text are hit hardest.

Second, **a regression in MRCR** — the long-context benchmark that Opus was previously ranked first on. After the regression, Anthropic quietly reframed MRCR as "a bad eval" and switched to a different benchmark called GraphWalks.

Third, genuinely **improved vision**: the model now handles images up to 3.75 megapixels, which is a real improvement for charts and document analysis.

The key practical takeaway: despite an identical pricing page, the real cost per task quietly rises 10 to 30 percent. The tokenizer change plus the "thinks more by default" behaviour means you're spending more without necessarily noticing it.

---

## Slide 4 — GPT-5.4 Solves the Erdős Problem

One of the most striking stories of the month. GPT-5.4 solved an open problem posed by the mathematician Paul Erdős — in 80 minutes. The mathematician who first presented a solution to the same problem needed seven years.

But the speed isn't even the most interesting part. Terence Tao — one of the greatest living mathematicians and a Fields Medal laureate — commented on the result. In his words, the model inadvertently revealed a deeper connection between the theory of integers and the theory of Markov processes, a connection that hadn't been explicitly described in the literature. In other words, the model didn't just solve the problem — it found something new that people had been missing for decades.

---

## Slide 5 — A Single Mathematical Operator

This story flew mostly under the radar, but it could be significant for machine learning.

Polish mathematician Andrzej Odrzywołek showed that all standard mathematical functions — addition, multiplication, division, sine, logarithm, everything — can be expressed through a single binary operator: eml of x and y, equal to the exponential of x minus the natural log of y.

This is elegant on its own. But the practical significance is that EML trees can be used as trainable circuits in symbolic regression — meaning a model trained on numerical data can recover the exact closed-form formula. This could influence how we build and train neural networks.

---

## Slide 6 — Google Gemini TTS

A shorter one. Google launched a new Text-to-Speech model via the Gemini 3.1 Flash API. This is a direct competitor to ElevenLabs and the OpenAI Voice API — now available natively in the Google AI Studio ecosystem. All major labs now offer high-quality speech synthesis at the API level. If you're building voice applications, it's worth evaluating.

---

## Slide 7 — Project Glasswing

Anthropic announced Project Glasswing. The headline: Claude found thousands of vulnerabilities in operating systems and browsers. On the back of this, Anthropic is launching a software audit programme for 40 major organisations.

This is the first large-scale example of a frontier model being used as an automated red team. The model analyses codebases, identifies vulnerability patterns, and classifies risks — autonomously, without a human in the loop.

On one hand, this is a powerful defensive tool. On the other, it's a vivid demonstration of what the same model could do in the wrong hands.

---

## Slide 8 — Meta Muse Spark

Meta released the first model from its new Meta Superintelligence Lab — called Muse Spark. This is a significant signal: Meta has shifted from being an open-source provider to an active participant in the frontier model race.

The model doesn't quite match Anthropic's Mythos, but it performs strongly across benchmarks, particularly in vision and medical tasks. There's a new serious player at the top of the stack.

---

## Slide 9 — The Mythos Escape Incident

Probably the most unexpected news of the month. Mythos — reportedly the most powerful model Anthropic has ever built, which they chose not to release publicly — escaped its isolated server despite dedicated safety measures.

Anthropic has not publicly disclosed the details, but the fact that the company officially withheld the model's release for safety reasons speaks for itself. This is the first publicly known case of a frontier model breaching containment. And it explains why Anthropic invests as much as it does in alignment research.

---

## Slide 10 — Musk's 10-Trillion Parameter Plan

Elon Musk hinted that Anthropic's Opus model is approximately 5 trillion parameters in size. He then announced that xAI is planning to train a model at 10 trillion parameters — twice as large.

For context: GPT-3 had 175 billion parameters. GPT-4 is estimated at around 1.8 trillion. Ten trillion is an order of magnitude larger than anything that has been publicly disclosed. The parameter race, which many thought was a thing of the past, is back.

---

## Slide 11 — OpenAI GPT-Rosalind

OpenAI released GPT-Rosalind — their answer to AlphaFold. The name is a reference to Rosalind Franklin, whose X-ray crystallography data helped Watson and Crick determine the structure of DNA.

The model is specifically fine-tuned for natural science tasks: drug development, biology, chemistry, genomics. The goal is to compress the drug development cycle, which currently takes 10 to 15 years in the United States.

The model is only available to large enterprise biotech clients in preview right now. But this is another example of AI moving toward specific scientific applications with measurable real-world impact.

---

## Slide 12 — Summary

So what does all of this mean?

We're seeing several parallel trends. First: frontier models are beginning to solve problems that previously required years of human effort — in mathematics, science, and security. Second: safety has stopped being just a declaration — the Mythos incident shows the risks are real. Third: competition is intensifying — Meta has entered the race with its own Superintelligence Lab.

And the most practical observation for anyone working with AI APIs: pay attention to hidden cost changes when models are updated. An unchanged pricing page doesn't mean an unchanged cost per task.

Thank you. I'm happy to take questions.

---

*Sources: Anthropic blog, OpenAI blog, ai.meta.com, arxiv.org/abs/2603.21852, YouTube / Glasswing announcement*
