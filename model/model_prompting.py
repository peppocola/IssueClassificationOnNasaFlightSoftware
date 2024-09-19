from abc import ABC, abstractmethod
from loguru import logger
import torch
from torch.cuda import OutOfMemoryError
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator
from accelerate.utils import gather_object
import json
import os
import time
from math import ceil
import wandb
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

class BaseLLMInference(ABC):
    def __init__(self, config):
        self.config = config
        self.logger = logger
        self.logger.add("logs/predict.log")
        self.responses = {}
        self.prompts = {}
        self.responses_path = None

    def load_prompts(self):
        with open(self.config["prompts_path"], 'r') as file:
            self.prompts = json.load(file)
        if self.responses:
            self.prompts = {k: v for k, v in self.prompts.items() if k not in self.responses}
        if len(prompts) < 1:
            raise Exception("No prompts to process. All prompts have already been processed.")
        self.logger.info(f"Loaded {len(self.prompts)} prompts")

    def load_responses(self):
        responses_dir = os.path.join(self.config["responses_dir"], self.config["model_name"])
        os.makedirs(responses_dir, exist_ok=True)
        self.responses_path = os.path.join(responses_dir, "responses.json")

        if os.path.exists(self.responses_path):
            with open(self.responses_path, 'r') as file:
                self.responses = json.load(file)
        else:
            self.responses = {}

    def save_responses(self):
        with open(self.responses_path, 'w') as file:
            json.dump(self.responses, file, indent=4)

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def process_prompts(self):
        pass

    def run(self):
        self.setup()
        self.load_responses()
        self.load_prompts()
        self.logger.info("Starting inference")
        self.process_prompts()
        self.logger.info("Inference complete")
        self.logger.info(f"Responses saved to {self.responses_path}")

class LLMInference(BaseLLMInference):
    def __init__(self, config):
        super().__init__(config)
        self.accelerator = None
        self.tokenizer = None
        self.model = None

    def setup(self):
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=self.config['load_in_4bit'],
            bnb_4bit_quant_type=self.config['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=self.config['bnb_4bit_use_double_quant'],
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.logger.info(f"Using model: {self.config['model_name']}")
        self.logger.info(f"Loading model and tokenizer")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.accelerator = Accelerator()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],    
            device_map={"": self.accelerator.process_index},
            torch_dtype=torch.bfloat16,
            quantization_config=nf4_config
        )

        self.logger.info(f"Model and tokenizer loaded")

    def process_prompts(self):
        self.accelerator.wait_for_everyone()
        start = time.time()

        save_every = self.config.get('save_every', 32)
        num_batches = ceil(len(self.prompts) / save_every)
        results = dict(outputs={}, num_tokens=0)
        keys = list(self.prompts.keys())

        for batch_idx in range(num_batches):
            start_idx = batch_idx * save_every
            end_idx = start_idx + save_every
            batch_ids = keys[start_idx:end_idx]
            batch_prompts = {k: self.prompts[k] for k in batch_ids}

            with self.accelerator.split_between_processes(batch_prompts) as prompts:
                for prompt_id, prompt in prompts.items():
                    conversation = []
                    if self.config.get('system_message'):
                        conversation.append({"role": "system", "content": prompt['system']})
                    conversation.append({"role": "user", "content": prompt['prompt']})
                    input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt").to("cuda")

                    if input_ids.shape[1] > self.config.get('max_input_token_length', 2048):
                        input_ids = input_ids[:, -self.config['max_input_token_length']:]

                    try:
                        generate_kwargs = {
                            "input_ids": input_ids,
                            "max_new_tokens": self.config.get('max_new_tokens', 500),
                            "do_sample": self.config.get('temperature', 0) != 0,
                            "eos_token_id": self.tokenizer.eos_token_id,
                        }

                        if generate_kwargs["do_sample"]:
                            generate_kwargs["temperature"] = self.config.get('temperature', 1.0)

                        output_tokenized = self.model.generate(**generate_kwargs)[0]
                        
                        output_tokenized = output_tokenized[len(input_ids[0]):]
                        results["outputs"][prompt_id] = self.tokenizer.decode(output_tokenized, skip_special_tokens=True)
                        results["num_tokens"] += len(output_tokenized)
                    except OutOfMemoryError as e:
                        print(f"skipping sample {prompt_id}")

                results = [results]

            results_gathered = gather_object(results)
            for prompt_id, response in results_gathered[0]['outputs'].items():
                self.responses[prompt_id] = response
            self.save_responses()

            results = dict(outputs={}, num_tokens=results_gathered[0]["num_tokens"])
            timediff = time.time() - start
            num_tokens = sum([r["num_tokens"] for r in results_gathered])

            self.logger.info(f"tokens/sec: {num_tokens // timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(self.prompts)}")
            
            # Log to wandb
            wandb.log({
                "tokens_per_second": num_tokens // timediff,
                "processing_time": timediff,
                "total_tokens": num_tokens,
                "processed_prompts": len(self.responses)
            })

        if self.accelerator.is_main_process:
            timediff = time.time() - start
            num_tokens = sum([r["num_tokens"] for r in results_gathered])
            self.logger.info(f"tokens/sec: {num_tokens // timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(self.prompts)}")
            
            # Final log to wandb
            wandb.log({
                "final_tokens_per_second": num_tokens // timediff,
                "total_processing_time": timediff,
                "final_total_tokens": num_tokens,
                "total_processed_prompts": len(self.responses)
            })

class OpenAILLMInference(BaseLLMInference):
    def __init__(self, config):
        super().__init__(config)
        self.client = None

    def setup(self):
        load_dotenv()
        self.logger.info(f"Setting up OpenAI client")
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=self.config['openai_organization'],
        )
        self.logger.info(f"OpenAI client set up")

    def process_prompts(self):
        start = time.time()
        total_tokens = 0

        for prompt_id, prompt in tqdm(self.prompts.items(), desc="Processing prompts"):
            self.logger.info(f"Processing prompt {prompt_id}")
            try:
                messages = []
                if self.config.get('system_message'):
                    messages.append({"role": "system", "content": self.config['system_message']})
                messages.append({"role": "user", "content": prompt['prompt']})

                response = self.client.chat.completions.create(
                    model=self.config["model_name"],
                    response_format={"type": "json_object"},
                    messages=messages,
                    temperature=self.config.get('temperature', 0),
                    max_tokens=self.config.get('max_new_tokens', 500),
                )
                
                text_response = response.choices[0].message.content
                json_response = json.loads(text_response)
                
                self.responses[prompt_id] = json_response
                total_tokens += response.usage.total_tokens

                self.save_responses()

                wandb.log({
                    "processed_prompts": len(self.responses),
                    "total_tokens": total_tokens,
                })

            except Exception as e:
                self.logger.error(f"Error processing prompt {prompt_id}: {str(e)}")

        timediff = time.time() - start
        self.logger.info(f"tokens/sec: {total_tokens // timediff}, time {timediff}, total tokens {total_tokens}, total prompts {len(self.prompts)}")
        
        wandb.log({
            "final_tokens_per_second": total_tokens // timediff,
            "total_processing_time": timediff,
            "final_total_tokens": total_tokens,
            "total_processed_prompts": len(self.responses)
        })

def process_llm_prompts(config):
    if config.get("use_openai", False):
        llm_inference = OpenAILLMInference(config)
    else:
        llm_inference = LLMInference(config)
    llm_inference.run()
