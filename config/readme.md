# Config

## DataArgs

- dataset_name
- overwrite_cache
- cache_file_name
- preprocessing_num_workers
- max_seq_length
- pad_to_max_length
- doc_stride
- max_answer_length
- train_retrieval
- eval_retrieval

## ModelArgs ( Reader, Retriver, Model, Tokenizer 관련 인자들 )

- model_name_or_path
- retriever_name
- reader_name
- config_name
- tokenizer_name

## TrainigArgs ( transformers에서 제공되는 인자들 )

- output_dir: str,
- overwrite_output_dir: bool = False,
- do_train: bool = False,
- do_eval: bool = None,
- do_predict: bool = False,
- evaluation_strategy: transformers.trainer_utils.IntervalStrategy = 'no',
- prediction_loss_only: bool = False,
- per_device_train_batch_size: int = 8,
- per_device_eval_batch_size: int = 8,
- per_gpu_train_batch_size: Union[int, NoneType] = None,
- per_gpu_eval_batch_size: Union[int, NoneType] = None,
- gradient_accumulation_steps: int = 1,
- eval_accumulation_steps: Union[int, NoneType] = None,
- learning_rate: float = 5e-05,
- weight_decay: float = 0.0,
- adam_beta1: float = 0.9,
- adam_beta2: float = 0.999,
- adam_epsilon: float = 1e-08,
- max_grad_norm: float = 1.0,
- num_train_epochs: float = 3.0,
- max_steps: int = -1,
- lr_scheduler_type: transformers.trainer_utils.SchedulerType = 'linear',
- warmup_ratio: float = 0.0,
- warmup_steps: int = 0,
- logging_dir: Union[str, NoneType] = <factory>,
- logging_strategy: transformers.trainer_utils.IntervalStrategy = 'steps',
- logging_first_step: bool = False,
- logging_steps: int = 500,
- save_strategy: transformers.trainer_utils.IntervalStrategy = 'steps',
- save_steps: int = 500,
- save_total_limit: Union[int, NoneType] = None,
- no_cuda: bool = False,
- seed: int = 42,
- fp16: bool = False,
- fp16_opt_level: str = 'O1',
- fp16_backend: str = 'auto',
- fp16_full_eval: bool = False,
- local_rank: int = -1,
- tpu_num_cores: Union[int, NoneType] = None,
- tpu_metrics_debug: bool = False,
- debug: bool = False,
- dataloader_drop_last: bool = False,
- eval_steps: int = None,
- dataloader_num_workers: int = 0,
- past_index: int = -1,
- run_name: Union[str, NoneType] = None,
- disable_tqdm: Union[bool, NoneType] = None,
- remove_unused_columns: Union[bool, NoneType] = True,
- label_names: Union[List[str], NoneType] = None,
- load_best_model_at_end: Union[bool, NoneType] = False,
- metric_for_best_model: Union[str, NoneType] = None,
- greater_is_better: Union[bool, NoneType] = None,
- ignore_data_skip: bool = False,
- sharded_ddp: str = '',
- deepspeed: Union[str, NoneType] = None,
- label_smoothing_factor: float = 0.0,
- adafactor: bool = False,
- group_by_length: bool = False,
- length_column_name: Union[str, NoneType] = 'length',
- report_to: Union[List[str], NoneType] = None,
- ddp_find_unused_parameters: Union[bool, NoneType] = None,
- dataloader_pin_memory: bool = True,
- skip_memory_metrics: bool = False,
- mp_parameters: str = '',
- **_(custom) pororo_prediction: bool = False_**
