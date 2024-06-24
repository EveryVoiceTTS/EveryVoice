# Should you build a TTS Model? What are the possible negative outcomes?

So, you're here because you want to build a TTS system - maybe for a language that doesn't have an existing one, but just because you *can* build a text-to-speech system, doesn't mean you *should*. The most important step of a new text-to-speech project is to consider what possible ethical problems could arise from the technology and which goals you are hoping to achieve with it. This section will walk you through some important questions to consider; you might also find that they apply broadly to other technology projects.

## Check Before you Tech!

Technology is flashy, and it seems like you can hardly turn a corner without someone talking about AI. However, as the excitment about the possibilities of this technology grow, so too have the cautionary warnings[^1][^2]. Amidst all the flurry of activity and hype - the fundamental question of *why are we building technology X, Y, or Z* should hopefully come up. What goals are we hoping to achieve, and what new problems might we be introducing with a new technology?

The following sections provide a couple of questions based on the excellent ["Check Before you Tech" guide](https://fpcc.ca/resource/check-before-you-tech/) for choosing language technology in a language revitalization context. While the original guide is geared towards technology users, we target our questions to technology developers and researchers. We urge you to consider these questions before beginning your TTS project.

!!! note
    This list is not intended to be a comprehensive list of all the ethical questions to consider, but rather a starting point for discussing and considering the impacts of the technology you are potentially creating.

#### Where is the data coming from? Do you have explicit permission from the creator of the data, and the speaker?

It is **not** ethical to build a TTS system with data that you do not have permission to use. You should not scrape or re-purpose data that you find online to build TTS systems unless the data comes with explicit permissions to do so.

For TTS, you are building a model of someone's likeness, so you should make sure that you have obtained permission from the data creator as well as the speaker whose likeness will be modeled. When permission is asked for, you should be clear with the person(s) about what the technology could be used for.

If you do not have enough time/resources to ask this question and obtain permission from all the relevant stakeholders, you should not build TTS models with the data.

#### What is your goal? How will TTS help you meet that goal?

As mentioned above, it's important to think about what you are actually trying to achieve with TTS. Not only will this help you determine whether EveryVoice TTS is the right toolkit for your application, but it will also help you determine whether you need to spend all the time and resources necessary to build a TTS system in the first place.

We invite you to consider whether your goal serving *you* or is it serving the people whose language you're working with? And if the answer is the latter, how do you know that, and how are you ensuring that that continues to be true? When discussing the project with relevant stakeholders, you should also mention any other goals you have in building this technology (e.g. publishing papers).

#### Where is the model going to be stored? Who has control and access to the model and who has ownership?

If the speaker or permissions-holders for the data or models change their mind about participation, how easy is it for them to stop the model? Do they have access to a 'kill switch'? Do they need to contact someone and make a request? Are there assurances about how long these requests will take to be processed?

Are there clear, agreed-upon guidelines for who has access to the model and data? Who maintains the control and access to these resources? In Canada, we encourage users to engage with the [First Nations Principles of OCAP®](https://fnigc.ca/ocap-training/) when planning a project.

#### What are the possible risks associated with this technology and how will I mitigate them?

Have you considered and discussed possible risks with the relevant stakeholders? Spend some time imagining ways that the tool could be misused, either by accidental or malicious actors. What if the model makes pronunciation mistakes? Will that embarrass the speaker? What if the model is made to say inappropriate things? What plans do you have to mitigate these risks?

Since this technology is relatively new, it can sometimes be hard to consider the ways that a technology can be misused, however we already see examples where TTS models are being used to generate fake news[^3]. Can you think of ways that similar so-called 'deep fakes' or impersonations could be used to cause harm?

[^1]: Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 2021. On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency (FAccT '21). Association for Computing Machinery, New York, NY, USA, 610–623. [https://doi.org/10.1145/3442188.3445922](https://doi.org/10.1145/3442188.3445922)
[^2]: Marie-Odile Junker. 2024. Data-mining and Extraction: the gold rush of AI on Indigenous Languages. In Proceedings of the Seventh Workshop on the Use of Computational Methods in the Study of Endangered Languages, pages 52–57, St. Julians, Malta. Association for Computational Linguistics. [https://aclanthology.org/2024.computel-1.8/](https://aclanthology.org/2024.computel-1.8/)
[^3]: [https://nypost.com/2024/06/14/us-news/michigan-gop-candidate-anthony-hudson-stands-by-ai-generated-mlk-jr-endorsement-video/](https://nypost.com/2024/06/14/us-news/michigan-gop-candidate-anthony-hudson-stands-by-ai-generated-mlk-jr-endorsement-video/)
