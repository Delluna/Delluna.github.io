import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 如果使用 GPU
    torch.cuda.manual_seed_all(seed)

    # 测试
    v_ids = [i for i in range(30)]
    set_seed(42)
    random.shuffle(v_ids) # 保证随机性
    print(v_ids)


# Instruct tuning
def instruct_tuning_LLM():
    # Import necessary libraries
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # login()  # Uncomment if you need to access private models
    # Load both base and instruct models for comparison
    instruct_model_name = "HuggingFaceTB/SmolLM3-3B"

    # Load tokenizers
    instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_name)

    # Load models (use smaller precision for memory efficiency)
    instruct_model = AutoModelForCausalLM.from_pretrained(
        instruct_model_name,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Models loaded successfully!")

    # Test SmolLM3's reasoning capabilities
    reasoning_prompts = [
        "What is 15 × 24? Show your work.",
        "A recipe calls for 2 cups of flour for 12 cookies. How much flour is needed for 30 cookies?",
        "If I have $50 and spend $18.75 on lunch and $12.30 on a book, how much money do I have left?"
    ]

    print("=== TESTING REASONING CAPABILITIES ===\n")

    for i, prompt in enumerate(reasoning_prompts, 1):
        print(f"Problem {i}: {prompt}")
        
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = instruct_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = instruct_tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = instruct_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,  # Lower temperature for more consistent reasoning
                do_sample=True,
                pad_token_id=instruct_tokenizer.eos_token_id
            )
            response = instruct_tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_start = response.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
            assistant_response = response[assistant_start:].split("<|im_end|>")[0]
            print(f"Answer: {assistant_response}")
        
        print("\n" + "-"*50 + "\n")


# VLM
def inference_with_VLM():
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from transformers.image_utils import load_image

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Quantization for efficiency
    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    model = AutoModelForImageTextToText.from_pretrained(model_name, quantization_config=quant_config).to(device)
    processor = AutoProcessor.from_pretrained(model_name)

    # Load image
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    image = load_image(image_url)

    # Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Can you describe the image?"}
            ]
        },
    ]

    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(device)

    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    # Extract only the assistant response
    assistant_response = generated_texts.split("Assistant:")[-1].strip()

    print(assistant_response)

# VLM batch 
def batch_infernece_with_VLM():
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from transformers.image_utils import load_image

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Quantization for efficiency
    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    model = AutoModelForImageTextToText.from_pretrained(model_name, quantization_config=quant_config).to(device)
    processor = AutoProcessor.from_pretrained(model_name)

    # 对于 decoder-only 模型+批量生成, 必须用左填充，否则生成结果会出错。

    # 右填充：内容靠左，填充靠右 → 适合读（编码器）
    # 左填充：填充靠左，内容靠右 → 适合写（解码器生成）

    # tokenizer默认右填充，因此在生成式大模型批量处理时需设置以下代码：
    processor.tokenizer.padding_side = 'left'

    # Load image
    image_url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    image1 = load_image(image_url1)

    # Create input messages
    messages1 = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Can you describe the image?"}
            ]
        },
    ]

    # Load image
    image_url2 = "https://img1.baidu.com/it/u=3722950138,788222029&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=500"
    image2 = load_image(image_url2)

    # Create input messages
    messages2 = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Can you describe the image?"}
            ]
        },
    ]



    # Prepare inputs
    prompt = processor.apply_chat_template([messages1, messages2], add_generation_prompt=True)
    print(f'prompt: {prompt}')
    inputs = processor(text=prompt, images=[[image1], [image2]], padding=True, return_tensors="pt")
    print(f'inputs.keys(): {list(inputs.keys())}')
    inputs = inputs.to(device)

    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    # # Extract only the assistant response
    for item in generated_texts:

    assistant_response = item.split("Assistant:")[-1].strip()

    print(assistant_response)


# training loop for fine-tune VLM using LoRA

def train_VLM_LoRA():
    # Import dependencies

    '''
    # Install required packages (run in Colab or your environment)
    
    pip install transformers datasets trl huggingface_hub trackio num2words==0.5.14
    '''
    

    import torch
    import os
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from transformers.image_utils import load_image

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_name)

    from datasets import load_dataset
    import matplotlib.pyplot as plt

    train_dataset, eval_dataset = load_dataset("HuggingFaceM4/ChartQA", split=["train[:10%]", "val[:10%]"])
    example = train_dataset[1]
    image = load_image(example["image"])

    print(example["query"])
    print(example["label"][0])

    system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
    Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
    The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
    Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

    def format_data(sample):
        return {
            "images": [sample["image"]],
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": sample["image"],
                        },
                        {
                            "type": "text",
                            "text": sample["query"],
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["label"][0]}],
                },
            ],
        }
    train_dataset = [format_data(sample) for sample in train_dataset]
    eval_dataset = [format_data(sample) for sample in eval_dataset]


    from peft import LoraConfig, get_peft_model

    # Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    # Apply PEFT model adaptation
    peft_model = get_peft_model(model, peft_config)

    # Print trainable parameters
    peft_model.print_trainable_parameters()

    from trl import SFTConfig, SFTTrainer

    # Configure training arguments using SFTConfig
    training_args = SFTConfig(
        output_dir="smol-course-smolvlm2-2.2b-instruct-trl-sft-ChartQA",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=25,
        save_strategy="steps",
        save_steps=25,
        optim="adamw_torch_fused",
        bf16=True,
        push_to_hub=True,
        report_to="trackio",
        max_length=None,
    )

    # Initialize the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Align the SFTTrainer params with your chosen dataset.

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(training_args.output_dir)



