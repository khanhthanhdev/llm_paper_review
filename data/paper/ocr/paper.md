# Sequence to Sequence Learning with Neural Networks

Ilya Sutskever Google ilyasu@google.com

Oriol Vinyals Google vinyals@google.com

Quoc V. Le   
Google   
qvl@google.com

# Abstract

Deep Neural Networks (DNNs) are powerful models that have achieved excellent performance on difficult learning tasks. Although DNNs work well whenever large labeled training sets are available, they cannot be used to map sequences to sequences. In this paper, we present a general end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure. Our method uses a multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector. Our main result is that on an English to French translation task from the WMT’14 dataset, the translations produced by the LSTM achieve a BLEU score of 34.8 on the entire test set, where the LSTM’s BLEU score was penalized on out-of-vocabulary words. Additionally, the LSTM did not have difficulty on long sentences. For comparison, a phrase-based SMT system achieves a BLEU score of 33.3 on the same dataset. When we used the LSTM to rerank the 1000 hypotheses produced by the aforementioned SMT system, its BLEU score increases to 36.5, which is close to the previous best result on this task. The LSTM also learned sensible phrase and sentence representations that are sensitive to word order and are relatively invariant to the active and the passive voice. Finally, we found that reversing the order of the words in all source sentences (but not target sentences) improved the LSTM’s performance markedly, because doing so introduced many short term dependencies between the source and the target sentence which made the optimization problem easier.

# 1 Introduction

Deep Neural Networks (DNNs) are extremely powerful machine learning models that achieve excellent performance on difficult problems such as speech recognition [13, 7] and visual object recognition [19, 6, 21, 20]. DNNs are powerful because they can perform arbitrary parallel computation for a modest number of steps. A surprising example of the power of DNNs is their ability to sort $N$ $N$ -bit numbers using only 2 hidden layers of quadratic size [27]. So, while neural networks are related to conventional statistical models, they learn an intricate computation. Furthermore, large DNNs can be trained with supervised backpropagation whenever the labeled training set has enough information to specify the network’s parameters. Thus, if there exists a parameter setting of a large DNN that achieves good results (for example, because humans can solve the task very rapidly), supervised backpropagation will find these parameters and solve the problem.

Despite their flexibility and power, DNNs can only be applied to problems whose inputs and targets can be sensibly encoded with vectors of fixed dimensionality. It is a significant limitation, since many important problems are best expressed with sequences whose lengths are not known a-priori. For example, speech recognition and machine translation are sequential problems. Likewise, question answering can also be seen as mapping a sequence of words representing the question to a sequence of words representing the answer. It is therefore clear that a domain-independent method that learns to map sequences to sequences would be useful.

Sequences pose a challenge for DNNs because they require that the dimensionality of the inputs and outputs is known and fixed. In this paper, we show that a straightforward application of the Long Short-Term Memory (LSTM) architecture [16] can solve general sequence to sequence problems. The idea is to use one LSTM to read the input sequence, one timestep at a time, to obtain large fixeddimensional vector representation, and then to use another LSTM to extract the output sequence from that vector (fig. 1). The second LSTM is essentially a recurrent neural network language model [28, 23, 30] except that it is conditioned on the input sequence. The LSTM’s ability to successfully learn on data with long range temporal dependencies makes it a natural choice for this application due to the considerable time lag between the inputs and their corresponding outputs (fig. 1).

There have been a number of related attempts to address the general sequence to sequence learning problem with neural networks. Our approach is closely related to Kalchbrenner and Blunsom [18] who were the first to map the entire input sentence to vector, and is related to Cho et al. [5] although the latter was used only for rescoring hypotheses produced by a phrase-based system. Graves [10] introduced a novel differentiable attention mechanism that allows neural networks to focus on different parts of their input, and an elegant variant of this idea was successfully applied to machine translation by Bahdanau et al. [2]. The Connectionist Sequence Classification is another popular technique for mapping sequences to sequences with neural networks, but it assumes a monotonic alignment between the inputs and the outputs [11].

![](images/62a6ba07779f9db2a6a6ddfa1076b4c9f3bbc18a5e54c3a93ec1e759f1d760bb.jpg)  
Figure 1: Our model reads an input sentence “ABC” and produces “WXYZ” as the output sentence. The model stops making predictions after outputting the end-of-sentence token. Note that the LSTM reads the input sentence in reverse, because doing so introduces many short term dependencies in the data that make the optimization problem much easier.

The main result of this work is the following. On the WMT’14 English to French translation task, we obtained a BLEU score of 34.81 by directly extracting translations from an ensemble of 5 deep LSTMs (with 384M parameters and 8,000 dimensional state each) using a simple left-to-right beamsearch decoder. This is by far the best result achieved by direct translation with large neural networks. For comparison, the BLEU score of an SMT baseline on this dataset is 33.30 [29]. The 34.81 BLEU score was achieved by an LSTM with a vocabulary of 80k words, so the score was penalized whenever the reference translation contained a word not covered by these $8 0 \mathrm { k }$ . This result shows that a relatively unoptimized small-vocabulary neural network architecture which has much room for improvement outperforms a phrase-based SMT system.

Finally, we used the LSTM to rescore the publicly available 1000-best lists of the SMT baseline on the same task [29]. By doing so, we obtained a BLEU score of 36.5, which improves the baseline by 3.2 BLEU points and is close to the previous best published result on this task (which is 37.0 [9]).

Surprisingly, the LSTM did not suffer on very long sentences, despite the recent experience of other researchers with related architectures [26]. We were able to do well on long sentences because we reversed the order of words in the source sentence but not the target sentences in the training and test set. By doing so, we introduced many short term dependencies that made the optimization problem much simpler (see sec. 2 and 3.3). As a result, SGD could learn LSTMs that had no trouble with long sentences. The simple trick of reversing the words in the source sentence is one of the key technical contributions of this work.

A useful property of the LSTM is that it learns to map an input sentence of variable length into a fixed-dimensional vector representation. Given that translations tend to be paraphrases of the source sentences, the translation objective encourages the LSTM to find sentence representations that capture their meaning, as sentences with similar meanings are close to each other while different sentences meanings will be far. A qualitative evaluation supports this claim, showing that our model is aware of word order and is fairly invariant to the active and passive voice.

# 2 The model

The Recurrent Neural Network (RNN) [31, 28] is a natural generalization of feedforward neural networks to sequences. Given a sequence of inputs $( x _ { 1 } , \dots , x _ { T } )$ , a standard RNN computes a sequence of outputs $( y _ { 1 } , \dots , y _ { T } )$ by iterating the following equation:

$$
\begin{array} { l l l } { { h _ { t } } } & { { = } } & { { \mathrm { s i g m } \left( { W ^ { \mathrm { h x } } } { x _ { t } } + { W ^ { \mathrm { h h } } } { h _ { t - 1 } } \right) } } \\ { { y _ { t } } } & { { = } } & { { { W ^ { \mathrm { y h } } } { h _ { t } } } } \end{array}
$$

The RNN can easily map sequences to sequences whenever the alignment between the inputs the outputs is known ahead of time. However, it is not clear how to apply an RNN to problems whose input and the output sequences have different lengths with complicated and non-monotonic relationships.

The simplest strategy for general sequence learning is to map the input sequence to a fixed-sized vector using one RNN, and then to map the vector to the target sequence with another RNN (this approach has also been taken by Cho et al. [5]). While it could work in principle since the RNN is provided with all the relevant information, it would be difficult to train the RNNs due to the resulting long term dependencies (figure 1) [14, 4, 16, 15]. However, the Long Short-Term Memory (LSTM) [16] is known to learn problems with long range temporal dependencies, so an LSTM may succeed in this setting.

The goal of the LSTM is to estimate the conditional probability $p ( y _ { 1 } , . . . , y _ { T ^ { \prime } } | x _ { 1 } , . . . , x _ { T } )$ where $( x _ { 1 } , \dots , x _ { T } )$ is an input sequence and $y _ { 1 } , \ldots , y _ { T ^ { \prime } }$ is its corresponding output sequence whose length $T ^ { \prime }$ may differ from $T$ . The LSTM computes this conditional probability by first obtaining the fixeddimensional representation $v$ of the input sequence $( x _ { 1 } , \dots , x _ { T } )$ given by the last hidden state of the LSTM, and then computing the probability of $y _ { 1 } , \ldots , y _ { T ^ { \prime } }$ with a standard LSTM-LM formulation whose initial hidden state is set to the representation $v$ of $x _ { 1 } , \ldots , x _ { T }$ :

$$
p ( y _ { 1 } , \dots , y _ { T ^ { \prime } } | x _ { 1 } , \dots , x _ { T } ) = \prod _ { t = 1 } ^ { T ^ { \prime } } p ( y _ { t } | v , y _ { 1 } , \dots , y _ { t - 1 } )
$$

In this equation, each $p ( y _ { t } | v , y _ { 1 } , \dots , y _ { t - 1 } )$ distribution is represented with a softmax over all the words in the vocabulary. We use the LSTM formulation from Graves [10]. Note that we require that each sentence ends with a special end-of-sentence symbol $\mathrm { ^ \bullet } { \ < } \mathrm { E O S } { > }$ , which enables the model to define a distribution over sequences of all possible lengths. The overall scheme is outlined in figure 1, where the shown LSTM computes the representation of “A”, “B”, “C”, $ { \mathrm {  ~ \cdot ~ } } \mathrm {  { ~ \cal { E } O S } }  { \mathrm {  ~ \cdot ~ } }$ and then uses this representation to compute the probability of “W”, “X”, “Y”, “Z”, “ $\mathrm { \partial \mathrm { \cdot } \mathrm { \angle E O S > \mathrm { \cdot } \mathrm { \ell } } }$ .

Our actual models differ from the above description in three important ways. First, we used two different LSTMs: one for the input sequence and another for the output sequence, because doing so increases the number model parameters at negligible computational cost and makes it natural to train the LSTM on multiple language pairs simultaneously [18]. Second, we found that deep LSTMs significantly outperformed shallow LSTMs, so we chose an LSTM with four layers. Third, we found it extremely valuable to reverse the order of the words of the input sentence. So for example, instead of mapping the sentence $a , b , c$ to the sentence $\alpha , \beta , \gamma$ , the LSTM is asked to map $c , b , a$ to $\alpha , \beta , \gamma$ , where $\alpha , \beta , \gamma$ is the translation of $a , b , c $ . This way, $a$ is in close proximity to $\alpha$ , $b$ is fairly close to $\beta$ , and so on, a fact that makes it easy for SGD to “establish communication” between the input and the output. We found this simple data transformation to greatly improve the performance of the LSTM.

# 3 Experiments

We applied our method to the WMT’14 English to French MT task in two ways. We used it to directly translate the input sentence without using a reference SMT system and we it to rescore the n-best lists of an SMT baseline. We report the accuracy of these translation methods, present sample translations, and visualize the resulting sentence representation.

# 3.1 Dataset details

We used the WMT’14 English to French dataset. We trained our models on a subset of 12M sentences consisting of 348M French words and 304M English words, which is a clean “selected” subset from [29]. We chose this translation task and this specific training set subset because of the public availability of a tokenized training and test set together with 1000-best lists from the baseline SMT [29].

As typical neural language models rely on a vector representation for each word, we used a fixed vocabulary for both languages. We used 160,000 of the most frequent words for the source language and 80,000 of the most frequent words for the target language. Every out-of-vocabulary word was replaced with a special “UNK” token.

# 3.2 Decoding and Rescoring

The core of our experiments involved training a large deep LSTM on many sentence pairs. We trained it by maximizing the log probability of a correct translation $T$ given the source sentence $S$ , so the training objective is

$$
1 / | S | \sum _ { ( T , S ) \in S } \log p ( T | S )
$$

where $s$ is the training set. Once training is complete, we produce translations by finding the most likely translation according to the LSTM:

$$
\hat { T } = \arg \operatorname* { m a x } _ { T } p ( T | S )
$$

We search for the most likely translation using a simple left-to-right beam search decoder which maintains a small number $B$ of partial hypotheses, where a partial hypothesis is a prefix of some translation. At each timestep we extend each partial hypothesis in the beam with every possible word in the vocabulary. This greatly increases the number of the hypotheses so we discard all but the $B$ most likely hypotheses according to the model’s log probability. As soon as the $ { \mathrm { \Sigma } } ^ { 6 6 } { < } \mathrm { E O S } > ^ { 5 }$ symbol is appended to a hypothesis, it is removed from the beam and is added to the set of complete hypotheses. While this decoder is approximate, it is simple to implement. Interestingly, our system performs well even with a beam size of 1, and a beam of size 2 provides most of the benefits of beam search (Table 1).

We also used the LSTM to rescore the 1000-best lists produced by the baseline system [29]. To rescore an n-best list, we computed the log probability of every hypothesis with our LSTM and took an even average with their score and the LSTM’s score.

# 3.3 Reversing the Source Sentences

While the LSTM is capable of solving problems with long term dependencies, we discovered that the LSTM learns much better when the source sentences are reversed (the target sentences are not reversed). By doing so, the LSTM’s test perplexity dropped from 5.8 to 4.7, and the test BLEU scores of its decoded translations increased from 25.9 to 30.6.

While we do not have a complete explanation to this phenomenon, we believe that it is caused by the introduction of many short term dependencies to the dataset. Normally, when we concatenate a source sentence with a target sentence, each word in the source sentence is far from its corresponding word in the target sentence. As a result, the problem has a large “minimal time lag” [17]. By reversing the words in the source sentence, the average distance between corresponding words in the source and target language is unchanged. However, the first few words in the source language are now very close to the first few words in the target language, so the problem’s minimal time lag is greatly reduced. Thus, backpropagation has an easier time “establishing communication” between the source sentence and the target sentence, which in turn results in substantially improved overall performance.

Initially, we believed that reversing the input sentences would only lead to more confident predictions in the early parts of the target sentence and to less confident predictions in the later parts. However, LSTMs trained on reversed source sentences did much better on long sentences than LSTMs trained on the raw source sentences (see sec. 3.7), which suggests that reversing the input sentences results in LSTMs with better memory utilization.

# 3.4 Training details

We found that the LSTM models are fairly easy to train. We used deep LSTMs with 4 layers, with 1000 cells at each layer and 1000 dimensional word embeddings, with an input vocabulary of 160,000 and an output vocabulary of 80,000. Thus the deep LSTM uses 8000 real numbers to represent a sentence. We found deep LSTMs to significantly outperform shallow LSTMs, where each additional layer reduced perplexity by nearly $1 0 \%$ , possibly due to their much larger hidden state. We used a naive softmax over 80,000 words at each output. The resulting LSTM has 384M parameters of which 64M are pure recurrent connections (32M for the “encoder” LSTM and 32M for the “decoder” LSTM). The complete training details are given below:

• We initialized all of the LSTM’s parameters with the uniform distribution between -0.08 and 0.08 We used stochastic gradient descent without momentum, with a fixed learning rate of 0.7. After 5 epochs, we begun halving the learning rate every half epoch. We trained our models for a total of 7.5 epochs.   
• We used batches of 128 sequences for the gradient and divided it the size of the batch (namely, 128).   
• Although LSTMs tend to not suffer from the vanishing gradient problem, they can have exploding gradients. Thus we enforced a hard constraint on the norm of the gradient [10, 25] by scaling it when its norm exceeded a threshold. For each training batch, we compute $s = \| g \| _ { 2 }$ , where $g$ is the gradient divided by 128. If $s > 5$ , we set $\textstyle g = { \frac { 5 g } { s } }$ . Different sentences have different lengths. Most sentences are short (e.g., length 20-30) but some sentences are long (e.g., length $> 1 0 0$ ), so a minibatch of 128 randomly chosen training sentences will have many short sentences and few long sentences, and as a result, much of the computation in the minibatch is wasted. To address this problem, we made sure that all sentences in a minibatch are roughly of the same length, yielding a $2 \mathbf { x }$ speedup.

# 3.5 Parallelization

A ${ \mathrm { C } } { + + }$ implementation of deep LSTM with the configuration from the previous section on a single GPU processes a speed of approximately 1,700 words per second. This was too slow for our purposes, so we parallelized our model using an 8-GPU machine. Each layer of the LSTM was executed on a different GPU and communicated its activations to the next GPU / layer as soon as they were computed. Our models have 4 layers of LSTMs, each of which resides on a separate GPU. The remaining 4 GPUs were used to parallelize the softmax, so each GPU was responsible for multiplying by a $1 0 0 0 \times 2 0 0 0 0$ matrix. The resulting implementation achieved a speed of 6,300 (both English and French) words per second with a minibatch size of 128. Training took about a ten days with this implementation.

# 3.6 Experimental Results

We used the cased BLEU score [24] to evaluate the quality of our translations. We computed our BLEU scores using multi-bleu. $\mathtt { p l } ^ { 1 }$ on the tokenized predictions and ground truth. This way of evaluating the BELU score is consistent with [5] and [2], and reproduces the 33.3 score of [29]. However, if we evaluate the best WMT’14 system [9] (whose predictions can be downloaded from statmt.org\matrix) in this manner, we get 37.0, which is greater than the 35.8 reported by statmt.org\matrix.

The results are presented in tables 1 and 2. Our best results are obtained with an ensemble of LSTMs that differ in their random initializations and in the random order of minibatches. While the decoded translations of the LSTM ensemble do not outperform the best WMT’14 system, it is the first time that a pure neural translation system outperforms a phrase-based SMT baseline on a large scale MT task by a sizeable margin, despite its inability to handle out-of-vocabulary words. The LSTM is within 0.5 BLEU points of the best WMT’14 result if it is used to rescore the 1000-best list of the baseline system.

Table 1: The performance of the LSTM on WMT’14 English to French test set (ntst14). Note that an ensemble of 5 LSTMs with a beam of size 2 is cheaper than of a single LSTM with a beam of size 12.   

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>test BLEU score (ntst14)</td></tr><tr><td rowspan=1 colspan=1>Bahdanau et al. [2]</td><td rowspan=1 colspan=1>28.45</td></tr><tr><td rowspan=1 colspan=1>Baseline System [29]</td><td rowspan=1 colspan=1>33.30</td></tr><tr><td rowspan=1 colspan=1>Single forward LSTM, beam size 12</td><td rowspan=1 colspan=1>26.17</td></tr><tr><td rowspan=1 colspan=1>Single reversed LSTM, beam size 12</td><td rowspan=1 colspan=1>30.59</td></tr><tr><td rowspan=1 colspan=1>Ensemble of 5 reversed LSTMs, beam size 1</td><td rowspan=1 colspan=1>33.00</td></tr><tr><td rowspan=1 colspan=1>Ensemble of 2 reversed LSTMs, beam size 12</td><td rowspan=1 colspan=1>33.27</td></tr><tr><td rowspan=1 colspan=1>Ensemble of 5 reversed LSTMs, beam size 2</td><td rowspan=1 colspan=1>34.50</td></tr><tr><td rowspan=1 colspan=1>Ensemble of 5 reversed LSTMs, beam size 12</td><td rowspan=1 colspan=1>34.81</td></tr></table>

Table 2: Methods that use neural networks together with an SMT system on the WMT’14 English to French test set (ntst14).   

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>test BLEU score (ntst14)</td></tr><tr><td rowspan=1 colspan=1>Baseline System [29]</td><td rowspan=1 colspan=1>33.30</td></tr><tr><td rowspan=1 colspan=1>Cho et al. [5]</td><td rowspan=1 colspan=1>34.54</td></tr><tr><td rowspan=1 colspan=1>Best WMT&#x27;14 result [9]</td><td rowspan=1 colspan=1>37.0</td></tr><tr><td rowspan=1 colspan=1>Rescoring the baseline 1000-best with a single forward LSTM</td><td rowspan=1 colspan=1>35.61</td></tr><tr><td rowspan=1 colspan=1>Rescoring the baseline 1000-best with a single reversed LSTM</td><td rowspan=1 colspan=1>35.85</td></tr><tr><td rowspan=1 colspan=1>Rescoring the baseline 1000-best with an ensemble of 5 reversed LSTMs</td><td rowspan=1 colspan=1>36.5</td></tr><tr><td rowspan=1 colspan=1>Oracle Rescoring of the Baseline 1000-best lists</td><td rowspan=1 colspan=1>~45</td></tr></table>

# 3.7 Performance on long sentences

We were surprised to discover that the LSTM did well on long sentences, which is shown quantitatively in figure 3. Table 3 presents several examples of long sentences and their translations.

# 3.8 Model Analysis

![](images/d87ff924656b71ea8e8cb34633a1d8b391de1362f449dfaf0662aa89cddf3d0f.jpg)  
Figure 2: The figure shows a 2-dimensional PCA projection of the LSTM hidden states that are obtained after processing the phrases in the figures. The phrases are clustered by meaning, which in these examples is primarily a function of word order, which would be difficult to capture with a bag-of-words model. Notice that both clusters have similar internal structure.

One of the attractive features of our model is its ability to turn a sequence of words into a vector of fixed dimensionality. Figure 2 visualizes some of the learned representations. The figure clearly shows that the representations are sensitive to the order of words, while being fairly insensitive to the

<table><tr><td rowspan=1 colspan=1>Type</td><td rowspan=1 colspan=1>Sentence</td></tr><tr><td rowspan=1 colspan=1>Our model</td><td rowspan=1 colspan=1>Ulrich UNK , membre du conseil d&#x27; administration du constructeur automobile Audi ,affirme qu&#x27; il s&#x27; agit d&#x27; une pratique courante depuis des années pour que les téléphonesportables puissent etre collectés avant les réunions du conseil d&#x27;administration afin qu&#x27; ilsne soient pas utilisés comme appareils d&#x27; écoute à distance .</td></tr><tr><td rowspan=1 colspan=1>Truth</td><td rowspan=1 colspan=1>Ulrich Hackenberg , membre du conseil d&#x27; administration du constructeur automobile Audi ,déclare que la collecte des téléphones portables avant les réunions du conseil , afin qu&#x27; ilsne puissent pas etre utilisés comme appareils d&#x27; écoute à distance , est une pratique courantedepuis des années .</td></tr><tr><td rowspan=1 colspan=1>Our model</td><td rowspan=1 colspan=1>&quot; Les téléphones cellulaires , qui sont vraiment une question , non seulement parce qu&#x27; ilspourraient potentiellement causer des interférences avec les appareils de navigation , maisnous savons , selon la FCC , qu&#x27; ils pourraient interférer avec les tours de téléphone cellulairelorsqu&#x27; ils sont dans I&#x27; air &quot; , dit UNK .</td></tr><tr><td rowspan=1 colspan=1>Truth</td><td rowspan=1 colspan=1>&quot; Les téléphones portables sont véritablement un problème , non seulement parce qu&#x27; ilspourraient éventuellement créer des interférences avec les instruments de navigation , maisparce que nous savons , d&#x27; après la FCC , qu&#x27; ils pourraient perturber les antennes-relais detéléphonie mobile s&#x27; ils sont utilisés à bord &quot;, a déclaré Rosenker .</td></tr><tr><td rowspan=1 colspan=1>Our model</td><td rowspan=1 colspan=1>Avec la crémation , il y a un &quot;&#x27; sentiment de violence contre le corps d&#x27; un etre cher &quot;qui sera &quot; réduit à une pile de cendres &quot; en très peu de temps au lieu d&#x27; un processus dedécomposition &quot; qui accompagnera les étapes du deuil &quot;.</td></tr><tr><td rowspan=1 colspan=1>Truth</td><td rowspan=1 colspan=1>Il y a , avec la crémation , &quot; une violence faite au corps aimé &quot;,qui va ètre &quot;réduit à un tas de cendres &quot;en très peu de temps , et non après un processus dedécomposition , qui &quot; accompagnerait les phases du deuil &quot;.</td></tr></table>

![](images/535f2d3b054d2beb2db0c701c27772117298fa96f029c5d54079e591621095ba.jpg)  
Table 3: A few examples of long translations produced by the LSTM alongside the ground truth translations. The reader can verify that the translations are sensible using Google translate.   
Figure 3: The left plot shows the performance of our system as a function of sentence length, where the $\mathbf { X }$ -axis corresponds to the test sentences sorted by their length and is marked by the actual sequence lengths. There is no degradation on sentences with less than 35 words, there is only a minor degradation on the longest sentences. The right plot shows the LSTM’s performance on sentences with progressively more rare words, where the $\mathbf { X }$ -axis corresponds to the test sentences sorted by their “average word frequency rank”.

replacement of an active voice with a passive voice. The two-dimensional projections are obtained using PCA.

# 4 Related work

There is a large body of work on applications of neural networks to machine translation. So far, the simplest and most effective way of applying an RNN-Language Model (RNNLM) [23] or a

Feedforward Neural Network Language Model (NNLM) [3] to an MT task is by rescoring the nbest lists of a strong MT baseline [22], which reliably improves translation quality.

More recently, researchers have begun to look into ways of including information about the source language into the NNLM. Examples of this work include Auli et al. [1], who combine an NNLM with a topic model of the input sentence, which improves rescoring performance. Devlin et al. [8] followed a similar approach, but they incorporated their NNLM into the decoder of an MT system and used the decoder’s alignment information to provide the NNLM with the most useful words in the input sentence. Their approach was highly successful and it achieved large improvements over their baseline.

Our work is closely related to Kalchbrenner and Blunsom [18], who were the first to map the input sentence into a vector and then back to a sentence, although they map sentences to vectors using convolutional neural networks, which lose the ordering of the words. Similarly to this work, Cho et al. [5] used an LSTM-like RNN architecture to map sentences into vectors and back, although their primary focus was on integrating their neural network into an SMT system. Bahdanau et al. [2] also attempted direct translations with a neural network that used an attention mechanism to overcome the poor performance on long sentences experienced by Cho et al. [5] and achieved encouraging results. Likewise, Pouget-Abadie et al. [26] attempted to address the memory problem of Cho et al. [5] by translating pieces of the source sentence in way that produces smooth translations, which is similar to a phrase-based approach. We suspect that they could achieve similar improvements by simply training their networks on reversed source sentences.

End-to-end training is also the focus of Hermann et al. [12], whose model represents the inputs and outputs by feedforward networks, and map them to similar points in space. However, their approach cannot generate translations directly: to get a translation, they need to do a look up for closest vector in the pre-computed database of sentences, or to rescore a sentence.

# 5 Conclusion

In this work, we showed that a large deep LSTM, that has a limited vocabulary and that makes almost no assumption about problem structure can outperform a standard SMT-based system whose vocabulary is unlimited on a large-scale MT task. The success of our simple LSTM-based approach on MT suggests that it should do well on many other sequence learning problems, provided they have enough training data.

We were surprised by the extent of the improvement obtained by reversing the words in the source sentences. We conclude that it is important to find a problem encoding that has the greatest number of short term dependencies, as they make the learning problem much simpler. In particular, while we were unable to train a standard RNN on the non-reversed translation problem (shown in fig. 1), we believe that a standard RNN should be easily trainable when the source sentences are reversed (although we did not verify it experimentally).

We were also surprised by the ability of the LSTM to correctly translate very long sentences. We were initially convinced that the LSTM would fail on long sentences due to its limited memory, and other researchers reported poor performance on long sentences with a model similar to ours [5, 2, 26]. And yet, LSTMs trained on the reversed dataset had little difficulty translating long sentences.

Most importantly, we demonstrated that a simple, straightforward and a relatively unoptimized approach can outperform an SMT system, so further work will likely lead to even greater translation accuracies. These results suggest that our approach will likely do well on other challenging sequence to sequence problems.

# 6 Acknowledgments

We thank Samy Bengio, Jeff Dean, Matthieu Devin, Geoffrey Hinton, Nal Kalchbrenner, Thang Luong, Wolfgang Macherey, Rajat Monga, Vincent Vanhoucke, Peng Xu, Wojciech Zaremba, and the Google Brain team for useful comments and discussions.

# References

[1] M. Auli, M. Galley, C. Quirk, and G. Zweig. Joint language and translation modeling with recurrent neural networks. In EMNLP, 2013.   
[2] D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.   
[3] Y. Bengio, R. Ducharme, P. Vincent, and C. Jauvin. A neural probabilistic language model. In Journal of Machine Learning Research, pages 1137–1155, 2003.   
[4] Y. Bengio, P. Simard, and P. Frasconi. Learning long-term dependencies with gradient descent is difficult. IEEE Transactions on Neural Networks, 5(2):157–166, 1994.   
[5] K. Cho, B. Merrienboer, C. Gulcehre, F. Bougares, H. Schwenk, and Y. Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In Arxiv preprint arXiv:1406.1078, 2014.   
[6] D. Ciresan, U. Meier, and J. Schmidhuber. Multi-column deep neural networks for image classification. In CVPR, 2012.   
[7] G. E. Dahl, D. Yu, L. Deng, and A. Acero. Context-dependent pre-trained deep neural networks for large vocabulary speech recognition. IEEE Transactions on Audio, Speech, and Language Processing - Special Issue on Deep Learning for Speech and Language Processing, 2012.   
[8] J. Devlin, R. Zbib, Z. Huang, T. Lamar, R. Schwartz, and J. Makhoul. Fast and robust neural network joint models for statistical machine translation. In ACL, 2014.   
[9] Nadir Durrani, Barry Haddow, Philipp Koehn, and Kenneth Heafield. Edinburgh’s phrase-based machine translation systems for wmt-14. In WMT, 2014.   
[10] A. Graves. Generating sequences with recurrent neural networks. In Arxiv preprint arXiv:1308.0850, 2013.   
[11] A. Graves, S. Fern´andez, F. Gomez, and J. Schmidhuber. Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In ICML, 2006.   
[12] K. M. Hermann and P. Blunsom. Multilingual distributed representations without word alignment. In ICLR, 2014.   
[13] G. Hinton, L. Deng, D. Yu, G. Dahl, A. Mohamed, N. Jaitly, A. Senior, V. Vanhoucke, P. Nguyen, T. Sainath, and B. Kingsbury. Deep neural networks for acoustic modeling in speech recognition. IEEE Signal Processing Magazine, 2012.   
[14] S. Hochreiter. Untersuchungen zu dynamischen neuronalen netzen. Master’s thesis, Institut fur Informatik, Technische Universitat, Munchen, 1991.   
[15] S. Hochreiter, Y. Bengio, P. Frasconi, and J. Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.   
[16] S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Computation, 1997.   
[17] S. Hochreiter and J. Schmidhuber. LSTM can solve hard long time lag problems. 1997.   
[18] N. Kalchbrenner and P. Blunsom. Recurrent continuous translation models. In EMNLP, 2013.   
[19] A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012.   
[20] Q.V. Le, M.A. Ranzato, R. Monga, M. Devin, K. Chen, G.S. Corrado, J. Dean, and A.Y. Ng. Building high-level features using large scale unsupervised learning. In ICML, 2012.   
[21] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 1998.   
[22] T. Mikolov. Statistical Language Models based on Neural Networks. PhD thesis, Brno University of Technology, 2012.   
[23] T. Mikolov, M. Karafi´at, L. Burget, J. Cernock\`y, and S. Khudanpur. Recurrent neural network based language model. In INTERSPEECH, pages 1045–1048, 2010.   
[24] K. Papineni, S. Roukos, T. Ward, and W. J. Zhu. BLEU: a method for automatic evaluation of machine translation. In ACL, 2002.   
[25] R. Pascanu, T. Mikolov, and Y. Bengio. On the difficulty of training recurrent neural networks. arXiv preprint arXiv:1211.5063, 2012.   
[26] J. Pouget-Abadie, D. Bahdanau, B. van Merrienboer, K. Cho, and Y. Bengio. Overcoming the curse of sentence length for neural machine translation using automatic segmentation. arXiv preprint arXiv:1409.1257, 2014.   
[27] A. Razborov. On small depth threshold circuits. In Proc. 3rd Scandinavian Workshop on Algorithm Theory, 1992.   
[28] D. Rumelhart, G. E. Hinton, and R. J. Williams. Learning representations by back-propagating errors. Nature, 323(6088):533–536, 1986.   
[29] H. Schwenk. University le mans. http://www-lium.univ-lemans.fr/˜schwenk/cslm_ joint_paper/, 2014. [Online; accessed 03-September-2014].   
[30] M. Sundermeyer, R. Schluter, and H. Ney. LSTM neural networks for language modeling. In INTERSPEECH, 2010.   
[31] P. Werbos. Backpropagation through time: what it does and how to do it. Proceedings of IEEE, 1990.