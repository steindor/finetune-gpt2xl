import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
# from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import torch
import wandb


import datasets
from datasets import load_dataset, Dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

os.environ["WANDB_DISABLED"] = "true"


def generate_cloze_sample(prompt, model, tokenizer, max_new_tokens=5):
    # Tokenize the input prompt
    input_tokens = tokenizer.encode(
        prompt, return_tensors='pt').to(model.device)

    # Replace the last word's tokens with the mask token
    last_word_start = len(input_tokens[0]) - 1
    while input_tokens[0, last_word_start] != tokenizer.sep_token_id and last_word_start > 0:
        last_word_start -= 1
    last_word_start += 1
    input_tokens[0, last_word_start:] = tokenizer.mask_token_id

    # Print the text with the mask token
    masked_text = tokenizer.decode(input_tokens[0], skip_special_tokens=False)

    # Pass the modified input to the model for generation
    max_length = len(input_tokens[0]) + max_new_tokens
    output = model.generate(
        input_tokens, max_length=max_length, num_return_sequences=1)

    # Print the sentence with the predicted token by the model
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(
        f"Prompt text: '{prompt}'\nPredicted output -> '{decoded_output}'")

    # print(f"Predicted words: '{decoded_output}'")

    return decoded_output


def generate_sample(prompt, model, tokenizer, max_new_tokens=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    max_length = len(input_ids[0]) + max_new_tokens
    output = model.generate(
        input_ids, max_length=max_length, num_return_sequences=1)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return f"'{decoded_output}'"


def generate_sample_and_cosine_similarity(prompt, model, tokenizer, actual_text, max_new_tokens=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    max_length = len(input_ids[0]) + max_new_tokens
    output = model.generate(
        input_ids, max_length=max_length, num_return_sequences=1)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Obtain the hidden states for the generated text
    generated_input_ids = tokenizer.encode(
        generated_text, return_tensors="pt").to(model.device)
    generated_output = model(generated_input_ids, output_hidden_states=True)
    generated_text_vector = generated_output.hidden_states[-1].mean(dim=1)

    # Obtain the hidden states for the actual text
    actual_input_ids = tokenizer.encode(
        actual_text, return_tensors="pt").to(model.device)
    actual_output = model(actual_input_ids, output_hidden_states=True)
    actual_text_vector = actual_output.hidden_states[-1].mean(dim=1)

    similarity = F.cosine_similarity(actual_text_vector, generated_text_vector)

    return generated_text, similarity.item()


def calculate_perplexity(actual_text, generated_text, model, tokenizer):
    # Convert the actual and generated text back to token ids.
    actual_ids = tokenizer.encode(actual_text, return_tensors='pt')
    generated_ids = tokenizer.encode(generated_text, return_tensors='pt')

    # Make sure the length of both tensors are the same for loss calculation
    min_length = min(actual_ids.size(1), generated_ids.size(1))
    actual_ids = actual_ids[:, :min_length]
    generated_ids = generated_ids[:, :min_length]

    # Move tensors to same device as model
    actual_ids = actual_ids.to(model.device)
    generated_ids = generated_ids.to(model.device)

    # Compute the loss.
    with torch.no_grad():
        logits = model(actual_ids).logits
        loss = F.cross_entropy(logits, generated_ids, reduction='mean')

    # Compute the perplexity.
    perplexity = torch.exp(loss)

    return perplexity.item()  # Return as a Python float


class SampleGenerationCallback(TrainerCallback):

    def __init__(self, prompts, train_dataset, tokenizer, model, data_args, text_table, run, log_steps, run_eval_on_step_count=10, num_samples=10):
        self.prompts = prompts
        self.train_dataset = train_dataset.select(range(num_samples))
        self.tokenizer = tokenizer
        self.model = model
        self.log_steps = log_steps
        self.data_args = data_args
        self.text_table = text_table
        self.run = run
        self.run_eval_on_step_count = run_eval_on_step_count

    def on_step_begin(self, args, state, control, **kwargs):

        transformers.utils.logging.set_verbosity_error()

        self.model.eval()

        if state.global_step % self.log_steps == 0 and state.global_step > 0:

            training_loss = None
            for log in reversed(state.log_history):
                if 'loss' in log:
                    training_loss = log['loss']
                    break
            if training_loss is not None:
                logger.info(
                    f"\n**********\nTraining loss at step {state.global_step}: {training_loss}\n**********\n")

                if self.data_args.use_wandb:
                    wandb.log({"Training Loss": training_loss,
                              "Step": state.global_step})

            for i, example in enumerate(self.train_dataset):
                # print(
                # f"\n{'*' * 10}\nTest sample {i+1} at step {state.global_step}:\n{'*' * 10}")
                full_text = self.tokenizer.decode(
                    example['input_ids'], skip_special_tokens=True)
                words = full_text.split()
                prompt_words = words[:30]
                prompt = ' '.join(prompt_words)
                # print(f"Prompt:\n{prompt}\n{'-' * 10}")
                actual_words = words[30:60]
                actual_text = ' '.join(actual_words)
                # print(f"Actual text:\n{actual_text}\n{'-' * 10}")
                generated_text, similarity = generate_sample_and_cosine_similarity(
                    prompt, self.model, self.tokenizer, actual_text)

                generated_text = generated_text[len(prompt):]

                # perplexity = calculate_perplexity(
                #     actual_text, generated_text, self.model, self.tokenizer)
                # print(f"Perplexity: {perplexity}")

                if self.data_args.use_wandb:
                    self.text_table.add_data(
                        i, state.global_step, prompt, actual_text, generated_text, similarity)

                    wandb.log({
                        "Prompt": prompt,
                        "Actual Text": actual_text,
                        "Generated Text": generated_text,
                        "Similarity": similarity,
                        "Step": state.global_step
                    })

        self.model.train()


prompts = [
    "Hvað er klukkan þegar það er kvöld?",
    "Hvaða stafur kemur á eftir F?",
    "Í hvernig nám fara læknar?",
    "Á hvaða ári var Ísland stofnað sem lýðveldi?",
    "Hann Tumi fer á fætur ",
    "Botninn er suður í ",
    "Maðurinn drakk kaffi með ",
    "Syngjandi sæll og ",
    "Lögreglan hélt manninum í gíslingu í þrjá ",
    "Þótt ótrulegt megi ",
    "Fólkið gengur á gangstéttum og bílarnir keyra á ",
    "Hundar gelta en kisur ",
    "Maðurinn brosti því hann var glaður, en drengurinn var með skeifu því hann var ",
    "Horfin eru sumarið og ",
    "Snjókorn og regn falla úr "
]

cloze_prompts = []

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0.dev0")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:

    """
        Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    do_eval: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to run eval on the validation set or not."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    use_wandb: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use wandb for logging."},
    )
    wandb_project: Optional[str] = field(
        default="Bloom-560m",
        metadata={"help": "Wandb project name."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(
        training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    '''
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name,
                                data_args.dataset_config_name)
        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
            )
            datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"

        datasets = load_dataset(
            extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
   '''

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    # Things that were changed from the huggingface file

    config.gradient_checkpointing = True
    config.use_cache = False

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if tokenizer.mask_token is None:
        tokenizer.mask_token = "[MASK]"
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(
            tokenizer.mask_token)

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    '''
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warn(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warn(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )'''

    # Dataset is already toknenized and shuffled
    lm_datasets = datasets.load_dataset(
        'stoddur/rmh_tokenized_512_train', num_proc=os.cpu_count())

    if training_args.do_train:
        '''
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        '''
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(
                range(data_args.max_train_samples))

    if training_args.do_eval:
        '''
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        '''
        eval_dataset = None  # lm_datasets["test"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(
                range(data_args.max_val_samples))

    if data_args.use_wandb:
        os.environ["WANDB_DISABLED"] = "false"
        if training_args.run_name == "finetuned":
            raise ValueError(
                "--run_name is required when using wandb. Please provide a name for your run."
            )

        logger.info("Using wandb")
        logger.info("Project: " + data_args.wandb_project)
        logger.info("Run name: " + training_args.run_name)
        logger.info(
            "Use flag --wandb_project to change project name within wandb")

        run = wandb.init(project=data_args.wandb_project,
                         name=training_args.run_name)
        wandb.config.update(training_args)
        text_table = wandb.Table(
            columns=["prompt_id", "Step", "Prompt", "Actual output", "Generated Text", "Cosine Similarity"])

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        callbacks=[SampleGenerationCallback(
            prompts,
            train_dataset,
            tokenizer,
            model,
            data_args,
            text_table if data_args.use_wandb else None,
            run if data_args.use_wandb else None,
            log_steps=training_args.logging_steps,
            run_eval_on_step_count=30,
            num_samples=10)]
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        perplexity = math.exp(metrics["eval_loss"])
        if data_args.use_wandb:
            wandb.log({"Perplexity": perplexity})
        metrics["perplexity"] = perplexity

        # import pdb
        # pdb.set_trace()

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if data_args.use_wandb:
        run.log({"Prompt table": text_table})
        wandb.finish()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
