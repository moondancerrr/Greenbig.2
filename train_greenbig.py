import gzip
import os
import json
import lightning
import signal
import sys
import torch
import transformers
from lightning.pytorch.plugins.environments import SLURMEnvironment
# Local imports.
import bigbro
import iterable

DEBUG = False

N_GPUS = 1 # 4 nodes with 4 GPUs.
N_ITEMS = 4_790_630
MAX_EPOCHS = 1
BATCH_SIZE = 2
ACC = 1

TRAIN_BATCHES = N_ITEMS * MAX_EPOCHS / BATCH_SIZE / N_GPUS
GRADIENT_STEPS = TRAIN_BATCHES / ACC

WARMUP = int(0.1 * GRADIENT_STEPS)
DECAY = GRADIENT_STEPS - WARMUP


GLOBAL_COUNTER = 0

fields = {'actinidia_chinensis': 0, 'aegilops_tauschii': 1, 'amborella_trichopoda': 2, 'ananas_comosus': 3, 'arabidopsis_halleri': 4, 'arabidopsis_lyrata': 5, 'arabidopsis_thaliana': 6, 'arabis_alpina': 7, 'asparagus_officinalis': 8, 'beta_vulgaris': 9, 'brachypodium_distachyon': 10, 'brassica_napus': 11, 'brassica_oleracea': 12, 'brassica_rapa': 13, 'brassica_rapa_ro18': 14, 'camelina_sativa': 15, 'cannabis_sativa_female': 16, 'capsicum_annuum': 17, 'chara_braunii': 18, 'checkpoints': 19, 'chenopodium_quinoa': 20, 'chlamydomonas_reinhardtii': 21, 'chondrus_crispus': 22, 'citrullus_lanatus': 23, 'citrus_clementina': 24, 'coffea_canephora': 25, 'corchorus_capsularis': 26, 'corylus_avellana': 27, 'cucumis_melo': 28, 'cucumis_sativus': 29, 'cyanidioschyzon_merolae': 30, 'cynara_cardunculus': 31, 'daucus_carota': 32, 'dioscorea_rotundata': 33, 'eragrostis_curvula': 34, 'eragrostis_tef': 35, 'eucalyptus_grandis': 36, 'eutrema_salsugineum': 37, 'ficus_carica': 38, 'galdieria_sulphuraria': 39, 'glycine_max': 40, 'gossypium_raimondii': 41, 'helianthus_annuus': 42, 'hordeum_vulgare': 43, 'ipomoea_triloba': 44, 'juglans_regia': 45, 'kalanchoe_fedtschenkoi': 46, 'lactuca_sativa': 47, 'leersia_perrieri': 48, 'lupinus_angustifolius': 49, 'malus_domestica_golden': 50, 'manihot_esculenta': 51, 'marchantia_polymorpha': 52, 'medicago_truncatula': 53, 'musa_acuminata': 54, 'nicotiana_attenuata': 55, 'nymphaea_colorata': 56, 'Nymphaea_colorata': 57, 'olea_europaea': 58, 'olea_europaea_sylvestris': 59, 'oryza_barthii': 60, 'oryza_brachyantha': 61, 'oryza_glaberrima': 62, 'oryza_glumipatula': 63, 'oryza_indica': 64, 'oryza_longistaminata': 65, 'oryza_meridionalis': 66, 'oryza_nivara': 67, 'oryza_punctata': 68, 'oryza_rufipogon': 69, 'oryza_sativa': 70, 'ostreococcus_lucimarinus': 71, 'other_organisms': 72, 'panicum_hallii': 73, 'panicum_hallii_fil2': 74, 'physcomitrium_patens': 75, 'pistacia_vera': 76, 'populus_trichocarpa': 77, 'prunus_avium': 78, 'prunus_dulcis': 79, 'prunus_persica': 80, 'quercus_lobata': 81, 'rosa_chinensis': 82, 'saccharum_spontaneum': 83, 'selaginella_moellendorffii': 84, 'sesamum_indicum': 85, 'setaria_italica': 86, 'setaria_viridis': 87, 'solanum_lycopersicum': 88, 'solanum_tuberosum': 89, 'solanum_tuberosum_rh8903916': 90, 'sorghum_bicolor': 91, 'theobroma_cacao': 92, 'theobroma_cacao_criollo': 93, 'trifolium_pratense': 94, 'triticum_aestivum': 95, 'triticum_aestivum_arinalrfor': 96, 'triticum_aestivum_cadenza': 97, 'triticum_aestivum_claire': 98, 'triticum_aestivum_jagger': 99, 'triticum_aestivum_julius': 100, 'triticum_aestivum_lancer': 101, 'triticum_aestivum_landmark': 102, 'triticum_aestivum_mace': 103, 'triticum_aestivum_mattis': 104, 'triticum_aestivum_norin61': 105, 'triticum_aestivum_paragon': 106, 'triticum_aestivum_robigus': 107, 'triticum_aestivum_stanley': 108, 'triticum_aestivum_weebil': 109, 'triticum_dicoccoides': 110, 'triticum_spelta': 111, 'triticum_turgidum': 112, 'triticum_urartu': 113, 'vigna_angularis': 114, 'vigna_radiata': 115, 'vitis_vinifera': 116, 'zea_mays': 117}

#fields = {
#    "arabidopsis_thaliana": 0,
#    "brassica_napus": 1,
#    "brassica_oleracea": 2,
#    "arabidopsis_halleri": 3,
#}

# Load the JSON dataset from a file

class TokenizerCollator:
   def __init__(self, tokenizer, fields, max_len=8192):
      self.tokenizer = tokenizer
      self.fields = fields
      self.max_len = max_len
   #def find_file_name(ex):
   #   return ex + '.txt'
   def __call__(self, examples):
      data = [json.load(open("./data" + ex + ".json")) for ex in examples]
      tokenized = tokenizer(
            [" ".join(ex["seq"].upper()) for ex in data],
            return_attention_mask=True,
            return_token_type_ids=True,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
      )
      
      L = tokenized["input_ids"].shape[-1]
      import pdb;pdb.set_trace()
      species_index = torch.tensor([[self.fields.get(ex["field"])] for ex in data])
      tokenized["token_type_ids"] = species_index.repeat(1, L)

      tokenized["labels"] = torch.tensor([
          [-100] + ex["calls"][:L-2] + [-100] + [-100]*(L-2-len(ex["calls"])) for ex in data
      ])

      return tokenized

class plTrainHarness(lightning.pytorch.LightningModule):
    "A Lightning train harness with AdamW, warmup and linear decay."

    def __init__(self, model, warmup=50, decay=1000000):
        super().__init__()
        self.model = model
        self.warmup = warmup
        self.decay = decay

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.trainer.model.parameters(),
              lr = 5e-5)
        warmup = torch.optim.lr_scheduler.LinearLR(
              optimizer,
              start_factor = 0.01,
              end_factor = 1., 
              total_iters = self.warmup)
        linear_decay = torch.optim.lr_scheduler.LinearLR(
              optimizer,
              start_factor = 1., 
              end_factor = 0.01,
              total_iters = self.decay)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
              optimizer = optimizer,
              schedulers = [warmup, linear_decay],
              milestones = [self.warmup]
        )   
        return  [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        #global GLOBAL_COUNTER
        #GLOBAL_COUNTER += 1
        #if GLOBAL_COUNTER == 20:
        #ids = batch.pop("id") # The ids are kept only for debugging.
        outputs = self.model(**batch)
        (current_lr,) = self.lr_schedulers().get_last_lr()
        info = { "loss": outputs.loss, "lr": current_lr }
        self.log_dict(dictionary=info, on_step = True, prog_bar = True)
        return outputs.loss


if __name__ == "__main__":

    torch.set_float32_matmul_precision("medium")

    tokenizer = transformers.PreTrainedTokenizerFast(
    tokenizer_file="./TokenizerModel/model.json",
         bos_token="[CLS]",
         eos_token="[SEP]",
         unk_token="[UNK]",
         sep_token="[SEP]",
         pad_token="[PAD]",
         cls_token="[CLS]",
         mask_token="[MASK]"
   )


   # The mighty BigBro model (BigBird with RoFormer position encoding).
   #config = transformers.BigBirdConfig(vocab_size=len(TokenizerCollator),
    config = transformers.BigBirdConfig(
         vocab_size=len(tokenizer),
         attention_type="block_sparse",
         max_position_embeddings=8192, 
         sep_token_id=2,
         type_vocab_size=119,
         # Config options for the RoFormer.
         embedding_size=768,
         rotary_value=False)
    model = bigbro.BigBroForTokenClassification(config=config)

    
    trash = bigbro.BigBroForTokenClassification.from_pretrained("./BigBroUnspliced/")
    #trash = bigbro.BigBroForTokenClassification.from_pretrained("./BigBroIntrons")
    state_dict = trash.state_dict()
    state_dict.pop("bert.embeddings.token_type_embeddings.weight")
    keys = model.load_state_dict(state_dict, strict=False)
    print(keys)
    del trash

    identifiers = [line.strip() for line in gzip.open('identifiers_data2.txt.gz','rt')]
    
    model.bert.set_attention_type("block_sparse")
    harnessed_model = plTrainHarness(model, warmup=WARMUP, decay=DECAY)

    from torch.utils.data import DataLoader
   
    data_loader = torch.utils.data.DataLoader(
         dataset = identifiers,
         collate_fn = TokenizerCollator(tokenizer, fields),
         batch_size = BATCH_SIZE,
         num_workers = 0 if DEBUG else 2,
         persistent_workers = False if DEBUG else True
    )   

    # Checkpoint and keep models every epoch. The default is only
    # one epoch so this is useless. Leaving it here in case the
    # number of epochs is increased.
    
    #from datetime import datetime, timedelta  #<---
    #import os                                 #<---
    
#    class EnhancedCheckpoint(pl.callbacks.ModelCheckpoint):
#       #def on_save_checkpoint(self, trainer, pl_module, checkpoint):
#       def __init__(self, train_time_interval):
#          self.train_time_interval = train_time_interval   #<---
#       def on_save_checkpoint(self, trainer, pl_module, train_time_interval, checkpoint):   #<---
#          model = trainer.model.module._forward_module.model
#          state_dict = model.state_dict()
#          #state_dict["self.config"] = model.config
#          torch.save(state_dict, "checkpointed_model.pt")
#
#    save_checkpoint = EnhancedCheckpoint(
#          dirpath = "checkpoints",
#          every_n_train_steps = 512,
#          train_time_interval=23             #<---
#    )   
    
    from datetime import time, timedelta
    save_checkpoint = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="my_awesome_checkpoint.pt",
        save_top_k=1,  # Save all checkpoints
        save_last=True,
        #every_n_train_steps = 2
        train_time_interval = timedelta(seconds = 5),  
    )
    trainer = lightning.pytorch.Trainer(
          default_root_dir = "checkpoints",
          strategy = lightning.pytorch.strategies.DDPStrategy(find_unused_parameters=True),
#          strategy = pl.strategies.FSDPStrategy(
##             cpu_offload = torch.distributed.fsdp.CPUOffload(offload_params=True),
#             activation_checkpointing = [
#                transformers.models.bert.modeling_bert.BertLayer,
#                transformers.models.bert.modeling_camembert.CamembertLayer,
#             ],
#             mixed_precision = torch.distributed.fsdp.MixedPrecision(
#                param_dtype=torch.bfloat16,
#                reduce_dtype=torch.bfloat16,
#                buffer_dtype=torch.bfloat16,
#             ),
#             auto_wrap_policy = wrapping_policy
#          ),
          accelerator = "gpu",
          devices = 1 if DEBUG else -1, 
          num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1)),
          accumulate_grad_batches = ACC,
#          max_epochs = MAX_EPOCHS,
          deterministic = False,
          precision = "bf16-mixed",
          # Options for a higher speed.
          enable_progress_bar = True, #if DEBUG else False,
          enable_model_summary = True if DEBUG else False,
          logger = True,
          # Checkpointing.
          enable_checkpointing = True,
          callbacks = [save_checkpoint],
          gradient_clip_val = 1.0,
          limit_train_batches = 1000,
    )   
     
    ckpt_path = None if len(sys.argv) < 2 else sys.argv[1]
    trainer.fit(harnessed_model, data_loader, ckpt_path=ckpt_path)
torch.save(model.state_dict(), 'output_greenbigpretrained.pt')

