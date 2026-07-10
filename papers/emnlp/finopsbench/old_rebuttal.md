Official Review of Submission3520 by Reviewer v7bm
Official Reviewby Reviewer v7bm20 Jun 2025, 11:00 (modified: 04 Aug 2025, 22:55)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer v7bm, Commitment ReadersRevisions
Paper Summary:
The paper studies prompt compression. The TPC framework is proposed, which generates a context-relevant task description using a task descriptor LM that is used for compression by comparing the context-relevant embedding to each sentence in the prompt. RL-based fine-tuning is applied to the task descriptor to capture the most relevant task descriptions.

Summary Of Strengths:
A lightweight Context-relevant Task Descriptor (CTD) model is designed to generate a task description that highlights the most relevant information within the prompt. Experiments on ZeroSCROLLS and LongBench show effectiveness of the proposed CTD.

Summary Of Weaknesses:
The authors claim that they propose a novel Task-agnostic Prompt Compression (TPC) framework that does not require input questions or templates. As far as I know, such frameworks already exist.

The method requires extra training/fine-tunning, which increases complexity. Thus, the comparison is not fair for training-free prompt compression method. The reported latency in Fig. 5 does not include time for training.

Comments Suggestions And Typos:
The complexity of the method (TPC-Base, TPC-Large, and TPC-Huge) should be discussed. Separate evaluations of training and inference latencies should be reported.

Confidence: 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
Soundness: 3.5
Excitement: 2.5
Overall Assessment: 2 = Resubmit next cycle: I think this paper needs substantial revisions that can be completed by the next ARR cycle.
Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 2 = They would be hard pressed to reproduce the results: The contribution depends on data that are simply not available outside the author's institution or consortium and/or not enough details are provided.
Datasets: 1 = No usable datasets submitted.
Software: 1 = No usable software released.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.




==>
Official Comment by Authors
Official Commentby Authors (Barys Liskavets, Shuvendu Roy, Ali Etemad, Shane K. Luke, +2 more)03 Jul 2025, 03:42 (modified: 04 Aug 2025, 22:55)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer v7bm, Commitment ReadersRevisions
Comment:
The authors claim that they propose a novel Task-agnostic Prompt Compression (TPC) framework that does not require input questions or templates. As far as I know, such frameworks already exist.

Yes, such methods exist. While most recent and SOTA compression methods are task-aware, there are a few task-agnostic methods, and we did not claim to be the first task-agnostic compression method. In fact, in Table 2, we have compared our method to existing task-agnostic compression methods.

The method requires extra training/fine-tunning, which increases complexity. Thus, the comparison is not fair for training-free prompt compression method. The reported latency in Fig. 5 does not include time for training. The complexity of the method (TPC-Base, TPC-Large, and TPC-Huge) should be discussed. Separate evaluations of training and inference latencies should be reported.

Kindly note that most existing SOTA compression methods, such as CPC (SOTA in the task-aware setup) and LLMLingua2 (SOTA in the task-agnostic setup), are training-required approaches. Since the primary goal of prompt compression is to reduce the inference time and computational cost during the deployment, existing methods (regardless of training-free or training-required), including ours, report and compare performance based on inference time (see Figure 5). No existing method has reported its training time.

Nonetheless, to further answer the question, we now report the number of parameters and training time in the table below. As shown, our training for TPC-Base took approximately 12 hours on a single Nvidia A100 GPU, which is higher than previous SOTA CPC, but is negligible in the broader context of training or fine-tuning large-scale LLMs.

Model	Parameters	Training L
TPC-base	0.5B	11.8 H
TPC-large	1B	20.3 H
TPC-Huge	7B	36.3 H
CPC	7B	5H


===========================
Official Review of Submission3520 by Reviewer FqbG
Official Reviewby Reviewer FqbG20 Jun 2025, 06:20 (modified: 04 Aug 2025, 22:55)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer FqbG, Commitment ReadersRevisions
Paper Summary:
The paper proposes Task-Agnostic Prompt Compression (TPC) to reduce the length of prompts for large language models (LLMs) without relying on task-specific questions or templates. TPC uses a Context-relevant Task Descriptor (CTD) to generate a task description from the prompt and a Context-aware Sentence Encoder (CSE) to score and select the most relevant sentences. CTD is refined using reinforcement learning guided by a novel reward function based on KL divergence between responses from full and compressed prompts. The paper introduces three model variants (Base, Large, Huge) and evaluates them across LongBench and ZeroSCROLLS, showing superior performance over state-of-the-art methods in both task-aware and task-agnostic settings.

Summary Of Strengths:
The paper introduces a novel compression approach that does not rely on explicit task instructions or handcrafted templates, enabling broader applicability across domains and input formats.
The paper uses a KL-divergence-based reinforcement learning signal to refine the task descriptor, which can ensure that compressed prompts yield outputs similar to full prompts, thus improving functional fidelity.
The authors conduct comprehensive empirical evaluations on LongBench and ZeroSCROLLS datasets, showing consistent improvements over strong baselines in both task-aware and task-agnostic setups, with detailed ablations to support design choices.
Summary Of Weaknesses:
While the proposed approach avoids explicit task templates, the task descriptor is trained on instruction-style prompts, making it implicitly task-aware; there is no rigorous evaluation on genuinely unknown or mixed-task inputs.
The paper does not include human evaluation or automatic metrics to assess whether the compressed prompts are logically coherent, fluent, or semantically complete, despite claiming improvements in compression quality.
The CTD and CSE modules are trained on datasets generated by prompting LLMs, which may introduce bias or lack diversity, potentially limiting TPC's generalization and robustness in real-world scenarios.
Comments Suggestions And Typos:
How does your approach handle prompts that do not align with typical instruction-following formats? Can you provide evidence that the model generalizes to truly unknown or mixed-task scenarios?
Have you conducted any human or automatic evaluations to verify that the compressed prompts are coherent and understandable? If not, how do you ensure the final prompt is suitable for downstream users?
Your KL-divergence reward assumes the original prompt's output is optimal. How do you account for hallucinations or suboptimal responses from the full prompt?
Given that the CTD and CSE modules are trained on LLM-generated synthetic datasets, how do you mitigate potential training artifacts or biases introduced by the prompt generation pipeline?
Confidence: 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
Soundness: 3.5
Excitement: 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
Overall Assessment: 3.5 = Borderline Conference
Best Paper Justification:
NA

Limitations And Societal Impact:
The Limitations section is very short and does not adequately discuss the limitations and societal impacts of the work.

Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 4 = They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.
Datasets: 1 = No usable datasets submitted.
Software: 1 = No usable software released.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.


==>
Official Comment by Authors
Official Commentby Authors (Barys Liskavets, Shuvendu Roy, Ali Etemad, Shane K. Luke, +2 more)03 Jul 2025, 03:41 (modified: 04 Aug 2025, 22:55)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer FqbG, Commitment ReadersRevisions
Comment:
How does your approach handle prompts that do not align with typical instruction-following formats? Can you provide evidence that the model generalizes to truly unknown or mixed-task scenarios? Have you conducted any human or automatic evaluations to verify that the compressed prompts are coherent and understandable? If not, how do you ensure the final prompt is suitable for downstream users?

To demonstrate the generalization capability of our proposed method, we conduct additional evaluations on OOD datasets and perform automatic assessments using an LLM as the judge. Specifically, we evaluate the compressed prompts along the following dimensions:

Coherency: We assess how consistent the compressed prompt is with the original version by asking an LLM to rate its coherence on a scale from 1 to 5.
Intent preservation: We evaluate how well the compressed prompt retains the user's original intent, as expressed in the uncompressed prompt.
Contextual information preservation: We assess the extent to which the compressed prompt enables the LLM to generate a response similar to what it would produce when given the original prompt.
The evaluations are conducted on the following out-of-distribution datasets:

a set of complex financial prompts (Financial News Prompts)
Self-Instruct, which includes complex prompts generated by LLM
P3, a subset of crowd-sourced instructions and demonstrations sampled from the Public Pool of Prompts (P3) dataset
UltraChat, a subset of Long instruction-following prompts curated for UltraLM. Examples include multi-paragraph instructions, simulated conversations, and educational tasks.
SystemCheck. A subset of the dataset containing generated user queries targeting 14 test-only system prompts with irrelevant in-context task demonstrations.
subset of OpenOrca, extended OpenAI prompts (via ShareGPT + FLAN), many with long, deeply nested reasoning tasks.
As shown in the table below, our method consistently achieves higher scores than existing approaches. Additionally, for each original instruction prompt from a given dataset, we directly ask the GPT-4.1 LLM to choose between the compressed prompt generated by our method and that produced by a competing method. We define the win rate of compressed prompt A over B as the number of times A is preferred over B, divided by the total number of samples from that dataset. A higher win rate thus reflects a more effective prompt compression strategy. According to the evaluation results, GPT-4.1 consistently prefers prompts compressed by our method over those produced by LLMLingua-2. We also observe a substantial performance advantage over other task-agnostic prompt compression baselines.

Comparison with LLMLingua-2:

Dataset	Method	Coherency	Intent Retainment	Contextual Information Retainment	LLM-as-a-Judge Winrate
Financial News Prompts	LLMLingua-2	2.5	2.5	1.6	0.6
Financial News Prompts	TPC	3.6	4.7	3.4	0.4
Self-Instruct	LLMLingua-2	2.2	2.5	1.8	0.2
Self-Instruct	TPC	3.6	2.8	2.4	0.8
P3	LLMLingua-2	3.2	3.6	2.9	0.4
P3	TPC	2.9	2.8	2.3	0.6
UltraChat	LLMLingua-2	2.4	2.9	2.3	0.1
UltraChat	TPC	3.7	3.4	2.7	0.9
SystemCheck	LLMLingua-2	1.8	2.8	2.4	0.1
SystemCheck	TPC	3.6	3.1	2.3	0.9
OpenOrca	LLMLingua-2	2.8	3.1	2.5	0.1
OpenOrca	TPC	3.4	3.3	2.4	0.9
Comparison with SelectiveContext:

Dataset	Method	Coherency	Intent Retainment	Contextual Information Retainment	LLM-as-a-Judge Winrate
Financial News Prompts	SelectiveContext	2.6	2.6	1.8	0.1
Financial News Prompts	TPC	3.6	4.7	3.4	0.9
Self-Instruct	SelectiveContext	2.7	3.1	2.1	0.1
Self-Instruct	TPC	3.6	2.8	2.4	0.9
P3	SelectiveContext	3.2	3.1	2.7	0.1
P3	TPC	2.9	2.8	2.3	0.9
UltraChat	SelectiveContext	2.3	2.4	1.7	0.0
UltraChat	TPC	3.7	3.4	2.7	1.0
SystemCheck	SelectiveContext	1.9	2.4	1.6	0.0
SystemCheck	TPC	3.6	3.1	2.3	1.0
OpenOrca	SelectiveContext	2.7	3.0	2.3	0.2
OpenOrca	TPC	3.4	3.3	2.4	0.8
Comparison with LLMLingua-1:

Dataset	Method	Coherency	Intent Retainment	Contextual Information Retainment	LLM-as-a-Judge Winrate
Financial News Prompts	LLMLingua-1	1.7	1.6	1.2	0.0
Financial News Prompts	TPC	3.6	4.7	3.4	1.0
Self-Instruct	LLMLingua-1	2.2	2.4	2.0	0.0
Self-Instruct	TPC	3.6	2.8	2.4	1.0
P3	LLMLingua-1	1.9	2.6	1.8	0.0
P3	TPC	2.9	2.8	2.3	1.0
UltraChat	LLMLingua-1	1.5	1.7	1.3	0.0
UltraChat	TPC	3.7	3.4	2.7	1.0
SystemCheck	LLMLingua-1	1.5	1.6	1.5	0.0
SystemCheck	TPC	3.6	3.1	2.3	1.0
OpenOrca	LLMLingua-1	1.8	2.2	1.7	0.0
OpenOrca	TPC	3.4	3.3	2.4	1.0


==>
Official Comment by Authors
Official Commentby Authors (Barys Liskavets, Shuvendu Roy, Ali Etemad, Shane K. Luke, +2 more)03 Jul 2025, 03:41 (modified: 04 Aug 2025, 22:55)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer FqbG, Commitment ReadersRevisions
Comment:
Your KL-divergence reward assumes the original prompt's output is optimal. How do you account for hallucinations or suboptimal responses from the full prompt?

While our KL-divergence reward compares compressed and full-prompt outputs, we acknowledge that full prompts may occasionally produce hallucinations or suboptimal responses. To mitigate this, we performed heuristic manual cleaning of the dataset before the RL sampling stage. Additionally, our method evaluates the relative preservation of the original semantics rather than assuming correctness. Specifically, the KL reward encourages the compressed prompt to yield a response distribution close to that of the original, without treating the original as ground truth.

The CTD and CSE modules are trained on datasets generated by prompting LLMs, which may introduce bias or lack diversity, potentially limiting TPC's generalization and robustness in real-world scenarios.

To reduce bias or lack of diversity, we curated our datasets from multiple instruction-tuned sources and applied multi-stage prompting to enhance coverage and structure. Moreover, our reinforcement learning refinement step helps align the task descriptor with downstream performance, serving as a practical safeguard against overfitting to synthetic artifacts. In response to the previous question, we show OOD evaluation, which confirms the strong generalization of our method.


==>
Official Comment by Reviewer FqbG
Official Commentby Reviewer FqbG03 Jul 2025, 11:17 (modified: 04 Aug 2025, 22:55)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer FqbG, Commitment ReadersRevisions
Comment:
Thank you to the authors for the detailed rebuttal. While several of my initial concerns have been addressed, a few important questions remain unresolved:

The use of an LLM as a judge is intriguing. Could you provide a comparison illustrating how its evaluations align with those of human judges? This would help assess the reliability of the LLM-based evaluation.

Regarding the heuristic manual cleaning of the dataset: Could you clarify the exact procedure followed and explain how this process effectively resolves the issues previously mentioned?

I apologize for the delayed response to your rebuttal. I hope you can clarify my concerns.


==>
Official Comment by Authors
Official Commentby Authors (Barys Liskavets, Shuvendu Roy, Ali Etemad, Shane K. Luke, +2 more)04 Jul 2025, 01:14 (modified: 04 Aug 2025, 22:55)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer FqbG, Commitment ReadersRevisions
Comment:
The use of an LLM as a judge is intriguing. Could you provide a comparison illustrating how its evaluations align with those of human judges? This would help assess the reliability of the LLM-based evaluation.

Although we cannot provide a statistically validated assessment of the agreement between LLM-as-a-judge and human judges due to time constraints, we are conducting several tests to evaluate the consistency of LLM-as-a-judge's assessment criteria with respect to the prompt’s compression rate.

In the absence of human ratings, we try to assess how much the ratings of each aspect obtained by the LLM judge change when we directly try to influence this aspect. To this end, we select a fixed set of lengthy prompts from different domains for our experiments. We conduct several evaluations: First, we estimate how the LLM judge's scores depend on the amount of information stored in the prompt. To do this, we compress a group of prompts at varying compression ratios and observe how the LLM judge score changes. We use TPC-Large and LLMLingua-2 for prompt compression. Despite using different compression strategies, we observe a clear correlation between LLM judge scores and compression ratios for both methods.

LLM-as-a-judge grade correlation with compression rate – Criterion: contextual_information_retainment

compression_ratio	0.2	0.4	0.6	0.8	1.0
TPC	3.714	4.0	4.429	4.714	5.0
LLMLingua-2	2.429	3.143	4.0	4.857	5.0
LLM-as-a-judge grade correlation with compression rate – Criterion: coherency

compression_ratio	0.2	0.4	0.6	0.8	1.0
TPC	3.286	3.571	4.0	4.429	5.0
LLMLingua-2	1.714	2.0	2.0	2.429	4.714
LLM-as-a-judge grade correlation with compression rate – Criterion: intent_retention

compression_ratio	0.2	0.4	0.6	0.8	1.0
TPC	4.286	4.857	5.0	5.0	5.0
LLMLingua-2	2.571	3.286	4.286	4.714	5.0
Next, we estimate how the intent_retention criterion is affected by the distortion of keywords in the user's query within long prompts. We ask LLM (GPT-4.1) to linearly distort these keywords and measure the resulting intent_retention scores given by LLM-as-a-judge.

LLM-as-a-judge intent_retention criterion correlation with intent obfuscation ratio

compression_ratio	0.0	0.2	0.5	0.8	1.0
obfuscation_severity_level	1.429	1.714	3.286	3.571	4.571
Then, we evaluate changes in the coherency score as we corrupt the original prompt. We linearly obfuscate a set of prompts by adding, replacing, or removing words or symbols with a given probability p. The results are shown below:

LLM-as-a-judge coherency criterion correlation with text structure obfuscation ratio

compression_ratio	0.0	0.02	0.05	0.08	0.1
obfuscation_severity_level	2.286	2.714	2.857	2.857	4.286
To check consistency in LLM judge win-rate ratings, we visually assess the reasonableness of LLM’s evaluations. Although we lack a large human evaluation team, we estimate ~90% agreement between the model’s evaluations and human preferences over 10 prompts.

We also replaced the LLM judge base model (GPT-4.1) with OpenAI's o3 model (a significantly stronger model) and observed 95% agreement between the two on a broader set of compressed prompts.


==>
Official Comment by Authors
Official Commentby Authors (Barys Liskavets, Shuvendu Roy, Ali Etemad, Shane K. Luke, +2 more)04 Jul 2025, 01:15 (modified: 04 Aug 2025, 22:55)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer FqbG, Commitment ReadersRevisions
Comment:
Regarding the heuristic manual cleaning of the dataset: Could you clarify the exact procedure followed and explain how this process effectively resolves the issues previously mentioned?

We use Llama-3.1-8B to generate responses from long and compressed prompts in the seed dataset. This allows us to obtain one generation per prompt in the initial dataset. Before training, we perform the following filtering procedure iteratively:

We use the all-mpnet-base-v2 model to generate semantic embeddings for all prompts and their corresponding generations in the initial dataset.

We randomly sample a subset of user prompts from the seed dataset.

We manually inspect instruction-response pairs for issues such as model refusals, incoherent or malformed outputs, ignoring context, or asking unnecessary follow-up questions when the answer is already inferable. We store such examples in a separate set.

For each prompt in a separate set, we use its embedding from step 1 to find the most similar prompts in the remaining dataset (using cosine similarity and a similarity threshold). These are added to a separate set too. Similarly, we use the embeddings of generations from a separate set to find the most similar generations in the rest of the dataset. These are also added to this set.

We exclude all samples from a separate set from the initial dataset.

We repeat the process from step 2.

We repeat the above steps until no malformed generations are found in several consecutive samples.

By removing semantically similar examples, we iteratively filter out subsets that cause poor model behavior. While we do not perform deep analysis for hallucinations in complex generations, this heuristic approach effectively removes the most obviously problematic examples from the training data.


==>
Official Comment by Reviewer FqbG
Official Commentby Reviewer FqbG04 Jul 2025, 09:21 (modified: 04 Aug 2025, 22:55)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer FqbG, Commitment ReadersRevisions
Comment:
I acknowledge that I have read the authors' rebuttal. Their responses have helped clarify several of my concerns. I have adjusted my score to reflect my updated assessment of the paper.



======================
Official Review of Submission3520 by Reviewer 7XAj
Official Reviewby Reviewer 7XAj19 Jun 2025, 11:22 (modified: 04 Aug 2025, 22:55)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 7XAj, Commitment ReadersRevisions
Paper Summary:
The paper presents a task-agnostic prompt compression framework that generates concise task descriptions and selects the most relevant sentences to efficiently shorten inputs for large language models, achieving or surpassing state of the art results without any task specific templates.

Summary Of Strengths:
TPC’s task agnostic design broadens its applicability across diverse tasks (e.g., summarization, QA, and code generation), outperforming existing methods on standard benchmarks.

Summary Of Weaknesses:
Limited Technical Novelty: The core components - sentence embedding for relevance scoring and task description generation - are relatively straightforward applications of existing techniques. The "task-agnostic" claim may be overstated since the method still relies on learning task patterns from training data, just without explicit questions at inference time.
Insufficient Experimental Analysis: The evaluation is limited to only two benchmark datasets (LongBench and ZeroSCROLLS), which may not adequately represent the diversity of real-world prompt compression scenarios.
Unclear Methodological Details: The reward function design for reinforcement learning training is not sufficiently detailed. How exactly does the reward capture "most relevant information" and what prevents the model from learning trivial compression strategies? The relationship between the Context-aware Sentence Embedding (CSE) and task descriptor components needs clearer explanation. The paper should include details on computational resource consumption during training, such as training duration, and total compute used.
Questionable Generalization Claims: While marketed as "task-agnostic," the method requires training on curated context-query pairs, which inherently introduces task-specific biases. The generalization to truly unseen task types remains questionable, and the paper doesn't demonstrate performance on domains significantly different from training data.
Computational Efficiency Concerns: The additional module for task description generation introduces extra inference steps, which lead to higher latency compared to simpler baselines. The authors acknowledge slower processing speed compared to competitive methods, but don't provide detailed computational analysis. The resource-intensive training process for the task descriptor module may limit practical adoption, especially given that simpler baselines might achieve comparable results.
Limited Baseline Comparisons: The paper primarily compares against LLMLingua variants but lacks comparison with other prompt compression approaches (eg DAC) or even simple heuristic methods that might perform surprisingly well on these tasks.
Evaluation Methodology Issues: The paper doesn't clearly explain how compression quality is measured beyond downstream task performance. There's insufficient analysis of what types of information are preserved versus discarded, and whether the compression maintains semantic coherence.
Scalability Questions: It's unclear how the method scales to very long prompts or diverse prompt structures beyond the evaluated benchmarks. The fixed sentence-level granularity may not be optimal for all prompt types.
Comments Suggestions And Typos:
The paper does not provide examples to illustrate potential failure cases, limiting insight into the method’s limitations.

Confidence: 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
Soundness: 3.5
Excitement: 2.5
Overall Assessment: 3 = Findings: I think this paper could be accepted to the Findings of the ACL.
Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 1 = They would not be able to reproduce the results here no matter how hard they tried.
Datasets: 1 = No usable datasets submitted.
Software: 1 = No usable software released.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Official Comment by Authors
Official Commentby Authors (Barys Liskavets, Shuvendu Roy, Ali Etemad, Shane K. Luke, +2 more)03 Jul 2025, 03:44 (modified: 04 Aug 2025, 22:55)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 7XAj, Commitment ReadersRevisions
Comment:
Limited Technical Novelty: The core components - sentence embedding for relevance scoring and task description generation - are relatively straightforward applications of existing techniques. The "task-agnostic" claim may be overstated since the method still relies on learning task patterns from training data, just without explicit questions at inference time.

While some individual components of TPC may bear conceptual similarities to existing techniques, our core technical novelty lies in their synergistic composition within the TPC framework, with a distinct focus on achieving task-agnostic prompt compression. Specifically, our innovation lies in the reward-guided fine-tuning of a task descriptor to dynamically generate context-relevant descriptions, coupled with its integration into a context-aware sentence embedding mechanism.

The evaluation is limited to only two benchmark datasets (LongBench and ZeroSCROLLS), which may not adequately represent the diversity of real-world prompt compression scenarios.

We followed the exact evaluation protocol adopted by all the existing literature on prompt compression, and reported the performance on LongBench and ZeroSCROLLS, for fair comparison. These are challenging benchmarks covering a diverse array of long-context understanding and generation tasks (e.g., QA, summarization, code). While not exhaustive, their comprehensive nature demonstrates TPC's strong performance, generalizability, and improvements over existing methods. Additionally, we provide evaluations on open out-of-domain datasets using LLM-as-a-Judge criteria. We present the details of these evaluations above.

Unclear Methodological Details: The reward function design for reinforcement learning training is not sufficiently detailed. How exactly does the reward capture "most relevant information" and what prevents the model from learning trivial compression strategies?

Our reward function is designed to align the compressed prompt’s informativeness with that of the original prompt by measuring the KL divergence between the response distributions of a pre-trained LLM given the original versus the compressed prompt (Eq. 4). This setup ensures that the generated task description captures the most relevant information necessary for downstream performance. To avoid trivial compression (e.g., selecting only generic sentences), we fine-tune the task descriptor via reinforcement learning using this reward signal, which penalizes deviations from the original response quality. Additionally, our ablation study (Table 3a) shows that removing this RL refinement significantly degrades performance, reinforcing its effectiveness in guiding non-trivial, task-relevant compression.

The relationship between the Context-aware Sentence Embedding (CSE) and task descriptor components needs clearer explanation.

As described in Sections 3.3 and 3.4, the task descriptor (CTD) generates a context-relevant task description, which is then used by the CSE to assess the relevance of each sentence in the prompt via embedding similarity. For clarity, we will revise the paper to better emphasize this relationship in the method overview section.

Questionable Generalization Claims: While marketed as "task-agnostic," the method requires training on curated context-query pairs, which inherently introduces task-specific biases. The generalization to truly unseen task types remains questionable, and the paper doesn't demonstrate performance on domains significantly different from training data.

Please refer to our first response to Reviewer FqbG, where we demonstrate that our proposed method exhibits strong generalization, including on out-of-distribution evaluation tasks.

Computational Efficiency Concerns: The additional module for task description generation introduces extra inference steps, which lead to higher latency compared to simpler baselines. The authors acknowledge slower processing speed compared to competitive methods, but don't provide detailed computational analysis. The resource-intensive training process for the task descriptor module may limit practical adoption, especially given that simpler baselines might achieve comparable results.

As discussed in Section 4.7 and illustrated in Figure 5, TPC exhibits slightly higher inference latency compared to some existing methods; however, it consistently outperforms them in terms of overall performance. To the best of our knowledge, no existing method achieves both better efficiency and higher performance than TPC. It is also worth noting that the primary goal of prompt compression methods is to reduce inference-time latency, and prior works did not report training-time latency. Nonetheless, we have included additional details regarding the training compute requirements in our response to Reviewer v7bm (see Response 2).


==>
Official Comment by Authors
Official Commentby Authors (Barys Liskavets, Shuvendu Roy, Ali Etemad, Shane K. Luke, +2 more)03 Jul 2025, 03:44 (modified: 04 Aug 2025, 22:55)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 7XAj, Commitment ReadersRevisions
Comment:
Limited Baseline Comparisons: The paper primarily compares against LLMLingua variants but lacks comparison with other prompt compression approaches (eg DAC) or even simple heuristic methods that might perform surprisingly well on these tasks.

To the best of our knowledge, we have included the most recent prompt compression approaches in Table 2. DAC is a concurrent work and has not yet been published. Nonetheless, we present the comparison to this method in the table below. As we find from this table, our method considerably outperforms DAC.

Method	NarrativeQA	Qasper	MultifieldQA	HotpotQA	2WikiMQA	Musique	GovReport	MultiNews	QMSum
DAC	24.85	33.46	40.12	42.37	39.57	22.44	30.4	25.77	25.95
TPC	23.72	37.75	50.25	45.06	53.87	26.78	31.15	25.45	21.82
Evaluation Methodology Issues: The paper doesn't clearly explain how compression quality is measured beyond downstream task performance. There's insufficient analysis of what types of information are preserved versus discarded, and whether the compression maintains semantic coherence.

We have now provided additional evaluation on the quality of compressed prompt in terms of coherence, intent preservation, and contextual information preservation using LLM as a judge. Please refer to the first response to Reviewer FqbG for more details.

Scalability Questions: It's unclear how the method scales to very long prompts or diverse prompt structures beyond the evaluated benchmarks. The fixed sentence-level granularity may not be optimal for all prompt types.

We agree that there may be situations where TPC does not perform as well. However, in our evaluations on benchmark datasets commonly used in prior prompt compression studies, we did not observe such issues. Consistent with previous work, we focus our evaluation on benchmarks that are sufficiently long, complex, and widely adopted in the literature. We acknowledge that evaluating TPC on extremely long prompts or inputs with highly diverse structures is an important direction, and we leave a more thorough analysis and comparison in such settings to future work.


==>
Comment
Official Commentby Reviewer 7XAj04 Jul 2025, 09:05 (modified: 04 Aug 2025, 22:55)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 7XAj, Commitment ReadersRevisions
Comment:
The author's response has addressed some of my concerns, so I have raised my score.
