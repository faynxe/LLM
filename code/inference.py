import itertools
import os
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import deepspeed
import torch
from djl_python import Input
from djl_python import Output
from djl_python.deepspeed import DeepSpeedService
from sagemaker_jumpstart_huggingface_script_utilities.djl_python.inference.textgeneration import format_djl_output
from sagemaker_jumpstart_huggingface_script_utilities.djl_python.inference.textgeneration import process_input
from sagemaker_jumpstart_huggingface_script_utilities.payload.dialog import convert_dialog_to_input_prompt
from sagemaker_jumpstart_huggingface_script_utilities.payload.enums import GenerationConfigParams
from sagemaker_jumpstart_huggingface_script_utilities.payload.stopping_criteria import (
    add_stopping_criteria_to_model_kwargs,
)
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import pipeline
from transformers.pipelines.text_generation import TextGenerationPipeline


MODEL_DIR_STR = "model_dir"
MODEL_ID_STR = "model_id"
TENSOR_PARALLEL_DEGREE_STR = "tensor_parallel_degree"
MAX_TOKENS_STR = "max_tokens"
MODEL_TYPE_STR = "model_type"
LOCAL_RANK_ENV_VAR_STR = "LOCAL_RANK"
TP_SIZE_STR = "tp_size"
AUTO_STR = "auto"
POST_DS_INFERENCE_INIT_STR = "post-ds-inference-init"
TEXT_GENERATION_STR = "text-generation"
GENERATED_TEXT_STR = "generated_text"
GENERATION_STR = "generation"
ACCEPT_EULA_STR = "accept_eula"
ACCEPT_EULA_ERROR_MESSAGE_STR = (
    "Need to pass custom_attributes='accept_eula=true' as part of header. This means you have read and accept the "
    "end-user license agreement (EULA) of the model. EULA can be found in model card description or from "
    "https://ai.meta.com/resources/models-and-libraries/llama-downloads/."
)


class ModelType(str, Enum):
    """Fine-tuning category for model weights."""

    BASE = "base"
    CHAT = "chat"


class Llama2Service(DeepSpeedService):
    """A service object for Llama 2 model family using the DJL Python engine."""

    def __init__(self) -> None:
        """Set initialization flag to False, model to be initialized upon invocation of initialize method."""
        self.initialized = False
        self.pipeline: Optional[TextGenerationPipeline] = None
        self.model_type: Optional[ModelType] = None

    def initialize(self, properties: Dict[str, Any]) -> None:
        """Load the model and initialize a transformers pipeline."""
        local_rank = int(os.getenv(LOCAL_RANK_ENV_VAR_STR, "0"))
        model_location = properties.get(MODEL_DIR_STR)
        if MODEL_ID_STR in properties:
            model_location = properties.get(MODEL_ID_STR)
        tensor_parallel = properties.get(TENSOR_PARALLEL_DEGREE_STR)
        max_tokens = properties.get(MAX_TOKENS_STR)
        self.model_type = ModelType(properties.get(MODEL_TYPE_STR))

        tokenizer = LlamaTokenizer.from_pretrained(model_location)

        model = LlamaForCausalLM.from_pretrained(
            model_location,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model = model.eval()
        torch.cuda.empty_cache()
        model = deepspeed.init_inference(
            model,
            tensor_parallel={TP_SIZE_STR: tensor_parallel},
            max_tokens=max_tokens,
            dtype=model.dtype,
            replace_method=AUTO_STR,
            replace_with_kernel_inject=True,
        )
        torch.cuda.empty_cache()
        deepspeed.runtime.utils.see_memory_usage(POST_DS_INFERENCE_INIT_STR, force=True)

        self.pipeline = pipeline(TEXT_GENERATION_STR, model=model.module, tokenizer=tokenizer, device=local_rank)

        self.initialized = True

    @format_djl_output
    def inference(self, inputs: Input) -> Output:
        """Define customized inference method to have hyperparameter validation for text generation task.

        Args:
            inputs (djl_python.inputs.Input): input containing payload and content type.
        Returns:
            outputs (djl_python.inputs.Output): model prediction output.
        """
        input_data, model_kwargs = process_input(inputs, input_data_as_list=False, use_parameters_key=True)
        model_kwargs = add_stopping_criteria_to_model_kwargs(model_kwargs, self.pipeline.tokenizer)

        if GenerationConfigParams.RETURN_FULL_TEXT not in model_kwargs:
            model_kwargs[GenerationConfigParams.RETURN_FULL_TEXT] = False

        if GenerationConfigParams.DO_SAMPLE not in model_kwargs:
            model_kwargs[GenerationConfigParams.DO_SAMPLE] = True

        if self.model_type == ModelType.CHAT:
            input_data = convert_dialog_to_input_prompt(input_data, self.pipeline.tokenizer)

        model_output: List[List[Dict[str, Any]]] = self.pipeline(input_data, **model_kwargs)
        model_output = list(itertools.chain(*model_output))
        for sequence in model_output:
            if self.model_type == ModelType.CHAT:
                sequence[GENERATION_STR] = {"role": "assistant", "content": sequence.pop(GENERATED_TEXT_STR)}
            else:
                sequence[GENERATION_STR] = sequence.pop(GENERATED_TEXT_STR)
        return model_output

    def handle(self, inputs: Input) -> Optional[Output]:
        """Handle an input query."""
        if self.initialized is False:
            self.initialize(inputs.get_properties())

        if inputs.is_empty():
            # Model server makes an empty call to warmup the model on startup
            return None

        properties = inputs.get_properties()
        eula = properties.get(ACCEPT_EULA_STR, "false").lower()
        if eula != "true":
            return Output().error(ACCEPT_EULA_ERROR_MESSAGE_STR)

        return self.inference(inputs)


_service = Llama2Service()
handle = _service.handle
