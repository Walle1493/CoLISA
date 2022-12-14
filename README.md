# CoLISA

CoLISA is a retriever-reader architecture for solving long-text multi-choice machine reading comprehension. It is based on the paper: ***CoLISA: Inner Interaction via Contrastive Learning for Multi-Choice Reading Comprehension***. Authors are [Mengxing Dong](https://github.com/Walle1493), [Bowei Zou](nlp.suda.edu.cn/~zoubowei), [Yanling Li](https://github.com/0308kkkk) and Yu Hong from Soochow University and Institute for Infocomm Research. The paper will be published in ECIR 2023 soon.

## Contents

- [Background](https://github.com/Walle1493/CoLISA#background)
- [Requirements](https://github.com/Walle1493/CoLISA#install)
- [Dataset](https://github.com/Walle1493/CoLISA#Dataset)
- [Train](https://github.com/Walle1493/CoLISA#Train)
- [Results](https://github.com/Walle1493/CoLISA#Results)
- [License](https://github.com/Walle1493/CoLISA#License)

## Background

Our idea is mainly inspired by the way humans solve multi-choice questions in reality: We first select relevant sentences from a long passage according to the given question and its multiple options to construct a short context; then we have multiple options interact within a specific question, in order to predict the final answer.

The goals for this repository are:

1. A **complete code** for QuALITY, and RACE. This repo offers an implementation for dealing with long-text multi-choice MRC dataset QuALITY, we also implement our method on another dataset RACE; you can also try this method on other datasets like DREAM yourself.
2. A comparison **description**. The performance on CoLISA has been listed in the paper.
3. A public space for **advice**. You are welcomed to propose an issue in this repo.

## Requirements

Clone this repo at your local server. Install necessary libraries listed below.

```bash
git clone git@github.com:Walle1493/CoLISA.git
pip install -r requirements.txt
```

You may install several libraries on yourself.

## Dataset

Take [QuALITY](https://github.com/nyu-mll/quality) ([click here seeing more](https://github.com/nyu-mll/quality)) for example, the dataset contains long articles, we utilize DPR to extract relevant sentences to shorten the whole passage. Then you need to convert such data in a RACE-like format. Our model is able to deal with RACE dataset. The processed data format is showed below:

```json
{
    "article_id": "22966",
    "set_unique_id": "22966_6AF3S2P3",
    "batch_num": "22",
    "writer_id": "1021",
    "source": "Gutenberg",
    "title": "Toy Shop",
    "year": 1968,
    "author": "Harrison, Harry",
    "topic": "Short stories; Science fiction; PS",
    "article": "The Atomic Wonder Space\n Wave Tapper hangs onto those space\n waves ...",
    "questions": [
        {
            "question": "What does the  Atomic Wonder Space Wave Tapper gadget do?",
            "question_unique_id": "22966_6AF3S2P3_1",
            "options": [
                "It can drive itself. ",
                "It levitates in the air.",
                "It flies in the air. ",
                "It can detect live in outer space. "
            ],
            "writer_label": 2,
            "gold_label": 2,
            "validation": [
                {
                    "untimed_annotator_id": "0015",
                    "untimed_answer": 2,
                    "untimed_eval1_answerability": 1,
                    "untimed_eval2_context": 2,
                    "untimed_best_distractor": 3
                },
                {...}
            ],
            "speed_validation": [
                {
                    "speed_annotator_id": "0010",
                    "speed_answer": 2
                },
                {...}
            ],
            "difficult": 0
        }
    ],
    "url": "http://aleph.gutenberg.org/2/2/9/6/22966//22966-h//22966-h.htm",
    "license": "This eBook is for the use of anyone anywhere in the United States and ..."
}
```

P.S.: You are supposed to make a change when dealing with other datasets like [DREAM](https://dataset.org/dream/), because the keys in json files are probably different from those in QuALITY and RACE.

## Train

The training step (including test module) depends mainly on several parameters. We trained our CoLISA model on 2 GPUs in 32G V100. Detailed hyper-parameters setup is showed below:

```bash
python run_colisa/run_colisa.py \
  --do_train \
  --do_eval \
  --do_test \
  --do_predict \
  --model_type deberta \
  --model_name_or_path microsoft/deberta-v3-large \
  --task_name quality \
  --data_dir ${DATA_DIR} \
  --train_file ${TRAIN_FILE} \
  --dev_file ${DEV_FILE} \
  --test_file ${TEST_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 32  \
  --learning_rate 1e-5 \
  --num_train_epochs 4 \
  --max_seq_length 512 \
  --alpha 0.5 \
  --beta 0.5 \
  --temperature 0.1 \
```

where `alpha` and `beta` stand for the ratio between main loss and contrastive loss, while `temperature` denotes the temperature coefficient in the contrastive learning formulation.

## Results

Our model achieved SOTA on the QuALITY dataset. You can visit the [leaderboard](https://nyu-mll.github.io/quality/) to see complete performance of all models on QuALITY. If possible, read our paper to suggest better options for us.

## License

[Soochow University](https://www.suda.edu.cn/) © [Mengxing Dong](https://github.com/Walle1493)

