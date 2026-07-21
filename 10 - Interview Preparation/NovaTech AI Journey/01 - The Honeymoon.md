# Phase 1: The Honeymoon
## Months 1–3

---

### Chapter 1: The Mandate

The email arrived on a Tuesday morning in January, subject line: *NovaTech 2.0 — AI First.*

Anika Sharma read it twice, then a third time. The CEO had declared that NovaTech, a mid-market enterprise software company with 10,000 employees and exactly zero production AI systems, would become "AI-first" by the end of the fiscal year. Attached was a PDF with buzzwords — synergy, transformation, paradigm shift — and a deadline: assemble a team by Friday.

Anika had been a solutions architect for seven years. She knew enterprise software. She knew distributed systems. She knew that an email with "synergy" in the attachment was not a strategy.

On Thursday, she walked into conference room 3B and found her team.

Raj Mehta was already there, laptop open, terminal windows covering his screen. He was a senior backend engineer who'd spent the last two years building data pipelines. He looked up, grunted, and went back to typing.

"Raj doesn't do small talk," said a voice behind her. David Chen, a business analyst from the product division, was holding a notebook covered in sticky notes. "I counted seventeen different AI ideas from stakeholders this week. Seventeen. And they all want them by next quarter."

Elena Vasquez arrived last, carrying a stack of architecture diagrams. She specialized in reliability engineering and had once caused a VP to cry during a code review. "I read the mandate," she said, sitting down. "It mentions 'deploy' and 'scale' and 'production.' Has anyone here deployed a model to production before?"

Silence.

"Right," Elena said. "Then we should probably talk about what we don't know."

---

Marcus Johnson, the newly appointed VP of AI, opened the meeting. He was polished, energetic, and clearly reading from slides prepared by someone else.

"NovaTech processes 2 million documents per month," he began. "Invoices, contracts, support tickets, compliance forms. All manual. We hire 300 people just to read documents. The CEO wants that down to zero."

He clicked to the next slide. It showed a rocket ship.

"I need a plan by Monday."

David flipped through his notebook. "I've collected requests. Customer support wants auto-triage. Legal wants contract clause extraction. Compliance wants anomaly detection. Sales wants lead scoring. HR wants resume parsing. Operations wants—"

"Stop," Anika said. "We can't do all of this. We don't have the data, the infrastructure, or the team."

"We have a mandate," Marcus said.

"A mandate isn't a strategy."

Raj snorted. First sound he'd made all meeting.

Marcus's jaw tightened. "Then what do you propose?"

Anika stood up and walked to the whiteboard.

"Document processing," she said. "It's the one thing every department needs. If we build a system that extracts structured data from unstructured documents — invoices, contracts, tickets — it's a platform, not a feature. Every department benefits. And David's list just became one problem instead of seventeen."

David nodded slowly. "That's actually smart. One use case, multiple wins."

Elena pointed at the whiteboard. "And what does that system look like?"

Anika drew a box. "User uploads document. Model extracts fields. System validates and stores."

"What model?"

"Don't know yet. We'll figure it out."

Elena frowned. "And the infrastructure?"

"We'll figure that out too."

"We're going to figure it out," Elena repeated flatly. "That's the plan."

"We have two weeks to prototype," Anika said. "Let's start."

---

### Chapter 2: The Notebook

Raj worked best alone. He'd taken the document processing problem, found a labeled dataset of 50,000 invoices on Kaggle, and spent three days in a Jupyter notebook.

By day four, he had something.

He called the team to his desk. "Watch this."

He pasted an invoice image into a cell and ran the code. The model — a pre-trained LayoutLMv2 he'd fine-tuned — annotated every field: vendor name, invoice number, date, total amount, line items. In green boxes.

"95% accuracy on the test set," Raj said.

David leaned in. "That's ... incredible."

"It's a notebook," Elena said.

"Your point?"

"My point is that a notebook is not a system. How are you serving this? What's the API? What happens when the document looks different from your training data? What's the latency?"

Raj waved his hand. "Those are engineering problems. The model works."

"The model works in this environment," Elena said, tapping his monitor. "That's not the same as working in production."

Anika stepped in. "Elena has a point. But this is a prototype. Let's validate the approach before we optimize the infrastructure."

"What's the approach?" Elena asked.

Raj pulled up his code. "LayoutLMv2. It's a transformer designed for document understanding. It reads text and layout together — word positions, bounding boxes, reading order. It handles the spatial relationships."

"The architecture?"

"Encoder-only transformer. Pre-trained on millions of documents, fine-tuned on invoice data. It's about 400MB. Runs on a T4 GPU."

"And when production documents don't look like Kaggle invoices?"

Raj shrugged. "We'll fine-tune on real data."

"When?"

"When we have real data."

Elena looked at Anika. "I want a data pipeline before we deploy anything. I want to know where production documents come from, how they're formatted, what the distribution looks like, and how it differs from training."

"I want a lot of things too," Raj muttered.

**Concepts introduced:** Jupyter notebook workflow, pre-trained models, fine-tuning, LayoutLMv2 architecture, encoder-only transformers, evaluation on held-out test split, the gap between notebook accuracy and production performance.

---

Marcus approved the prototype demo. He brought the CEO's chief of staff. The demo worked perfectly on three carefully chosen documents.

"Amazing," the chief of staff said. "When can we use this?"

"Give us a month," Anika said.

Elena winced.

"A month," Marcus repeated, writing it down. "Excellent."

After everyone left, Elena cornered Anika.

"We're not ready. We don't have a data pipeline. We don't have monitoring. We don't even know where the production data lives."

"We'll build it as we go."

"That's not how reliability works. You can't bolt on production readiness after the model is deployed."

"We don't have the luxury of time, Elena. The CEO wants results. I'll give him results, and we'll fix the rest later."

"Later," Elena said, shaking her head. "Technical debt with interest."

---

### Chapter 3: The First Deployment

Elena spent the next three weeks building infrastructure while Raj iterated on the model.

She set up a Kubernetes cluster on AWS. EKS, Node groups with T4 GPUs. A FastAPI service wrapping Raj's model. An S3 bucket for document storage. A PostgreSQL database for extracted fields.

It was fragile. No monitoring. No rolling deployments. No autoscaling.

"Ship it," Marcus said.

"Wait," Elena said. "I haven't—"

"Ship. It."

They shipped it.

The first week went smoothly. 500 documents processed. 93% accuracy. Users were impressed. Marcus sent a victory email to the exec team.

Day 8, everything broke.

Volume jumped from 500 to 5,000 documents overnight. The single T4 GPU saturated. Latency went from 2 seconds to 45 seconds. New document formats appeared — handwritten invoices, multi-page contracts, scanned images with low resolution. The model started outputting gibberish. Confidence scores dropped from 95% to 60%.

Users started complaining. Then panicking. Then escalating.

"We have a problem," David said, walking into the war room.

"Define problem," Raj said, not looking up from his logs.

"Legal sent a contract through the system. It classified 'Total Amount' as 'Date.' They almost signed it."

Elena froze. "Please tell me no one actually signed that."

"We caught it. But we almost didn't."

Anika was pacing. "What's happening technically?"

Raj pulled up his notebook. "Multiple issues. One, the model hasn't seen multi-page layouts before. Two, handwritten text is out of distribution. Three, the image preprocessing pipeline — we assumed all documents were scanned at 300 DPI, but production documents vary wildly. Four—"

"Stop," Anika said. "What's the fix?"

"There's no single fix. We need more training data. We need to normalize inputs. We need a quality gate that rejects low-confidence predictions."

"How long?"

"A month. Minimum."

Marcus walked in. "The VP of Legal is on the phone. She wants to know how a system that was '95% accurate' could miss a five-digit field."

No one answered.

**Concepts introduced:** Production deployment on Kubernetes, serving infrastructure (FastAPI + GPU), real-world data distribution shift, out-of-distribution detection, image preprocessing pipeline inconsistencies, confidence calibration, quality gates, the gap between controlled demo and production chaos.

---

"Here's what we learned," Anika said at the post-mortem.

They were in the same conference room, three months after they started. The mood was different.

"One: our training data didn't match production data. We trained on clean Kaggle invoices and got hit with handwritten notes and multi-page contracts."

"Training-serving skew," Elena said.

"Two: we had no way to detect this. No monitoring, no drift detection, no quality gates. If the contract had been signed, we would have caused a real legal problem."

"Three: we prioritized speed over reliability. That's on me."

She looked at Elena. "You warned us. I should have listened."

Elena didn't say "I told you so." But it hung in the air.

"Where do we go from here?" David asked.

"Back to the drawing board," Anika said. "But this time, we're going to be honest about what we don't know."
