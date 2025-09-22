# Legacy Test file for config trackie, current in C++ TrackieLink.



import os
import sys 
import asyncio
import base64
import io
import json
import logging


dependency_path = r"C:\TrackieIntelligence\WorkTools\WorkingTools"


if dependency_path not in sys.path:
    sys.path.insert(0, dependency_path)
    print(f"'{dependency_path}' adicionado ao sys.path para importa��o de m�dulos.")
else:
    print(f"'{dependency_path}' j� est� no sys.path.")
# --- Fim da configura��o do sys.path ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
import traceback
import time
import argparse
import threading
from typing import Dict, Any, Optional, List, Tuple
from playsound import playsound


# Bibliotecas de Terceiros
import cv2
import pyaudio
from PIL import Image
import mss
import pandas as pd
from google import genai
from google.genai import types
from google.genai.types import Tool, FunctionDeclaration
from google.genai.types import Content, Part
from google.genai.types import GenerateContentConfig
from google.genai import errors
from google.genai.types import LiveConnectConfig, Modality
from google.protobuf.struct_pb2 import Value, Struct
from ultralytics import YOLO
import numpy as np
from deepface import DeepFace
import torch
import torchvision
import timm


# --- Constantes ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001"
DEFAULT_MODE = "camera"
BaseDir = "C:\TrackieIntelligence"
DANGER_SOUND_PATH = os.path.join(BaseDir, "WorkTools", "SoundBibTrackie", "Trackiedanger.wav")

# --- Inicializa��o do PyAudio ---
try:
    pya = pyaudio.PyAudio()
except Exception as e:
    logger.error(f"Erro ao inicializar PyAudio: {e}. O �udio n�o funcionar�.")
    pya = None

def play_wav_file_sync(filepath):
    """Reproduz um arquivo WAV de forma s�ncrona."""
    try:
        logger.info(f"Reproduzindo arquivo de �udio: {filepath}")
        playsound(filepath)
        logger.info(f"Reprodu��o de {filepath} conclu�da.")
    except Exception as e:
        logger.error(f"Erro ao reproduzir o arquivo WAV {filepath}: {e}")     

# --- Caminho para o arquivo de prompt ---
SYSTEM_INSTRUCTION_PATH = os.path.join(BaseDir,"UserSettings", "Prompt's", "TrckItcs.txt")
CONFIG_PATH = os.path.join(BaseDir, "UserSettings", "trckconfig.json")
#SNOWBOY_MODEL_PATH = os.path.join(BaseDir, "WorkTools", "trackie.pmdl")

# Carregar o JSON e extrair CFG'S
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    trckuser = config_data.get('trckuser', 'Usu�rio Padr�o')  # Valor padr�o se trckuser n�o existir
except FileNotFoundError:
    logging.error(f"Arquivo de configura��o n�o encontrado: {CONFIG_PATH}")
    trckuser = 'Usu�rio Padr�o'  # Valor padr�o em caso de erro
except json.JSONDecodeError as e:
    logging.error(f"Erro ao decodificar JSON em {CONFIG_PATH}: {e}")
    trckuser = 'Usu�rio Padr�o'
except Exception as e:
    logging.error(f"Erro ao carregar configura��o de {CONFIG_PATH}: {e}")
    trckuser = 'Usu�rio Padr�o'

# YOLO
YOLO_MODEL_PATH = os.path.join(BaseDir,"WorkTools", "yolov8n.pt")
DANGER_CLASSES = {
    # (Dicion�rio DANGER_CLASSES inalterado - omitido para brevidade)
    'faca':             ['knife'],
    'tesoura':          ['scissors'],
    'barbeador':        ['razor'],
    'serra':            ['saw'],
    'machado':          ['axe'],
    'machadinha':       ['hatchet'],
    'arma_de_fogo':     ['gun'],
    'pistola':          ['pistol'],
    'rifle':            ['rifle'],
    'espingarda':       ['shotgun'],
    'rev�lver':         ['revolver'],
    'bomba':            ['bomb'],
    'granada':          ['grenade'],
    'fogo':             ['fire'],
    'chama':            ['flame'],
    'fuma�a':           ['smoke'],
    'isqueiro':         ['lighter'],
    'f�sforos':         ['matches'],
    'fog�o':            ['stove'],
    'superf�cie_quente':['hot surface'],
    'vela':             ['candle'],
    'queimador':        ['burner'],
    'fio_energizado':   ['live_wire'],
    'tomada_el�trica':  ['electric_outlet'],
    'bateria':          ['battery'],
    'vidro_quebrado':   ['broken_glass'],
    'estilha�o':        ['shard'],
    'agulha':           ['needle'],
    'seringa':         ['syringe'],
    'martelo':          ['hammer'],
    'chave_de_fenda':   ['wrench'],
    'furadeira':        ['drill'],
    'motosserra':       ['chainsaw'],
    'carro':            ['car'],
    'motocicleta':      ['motorcycle'],
    'bicicleta':        ['bicycle'],
    'caminh�o':         ['truck'],
    '�nibus':           ['bus'],
    'urso':             ['bear'],
    'cobra':            ['snake'],
    'aranha':           ['spider'],
    'jacar�':           ['alligator'],
    'penhasco':         ['cliff'],
    'buraco':           ['hole'],
    'escada':           ['stairs'],
}
YOLO_CONFIDENCE_THRESHOLD = 0.40
YOLO_CLASS_MAP = {
    # (Dicion�rio YOLO_CLASS_MAP inalterado - omitido para brevidade)
    "pessoa":                     ["person"],
    "gato":                       ["cat"],
    "cachorro":                   ["dog"],
    "coelho":                     ["rabbit"],
    "urso":                       ["bear"],
    "elefante":                   ["elephant"],
    "zebra":                      ["zebra"],
    "girafa":                     ["giraffe"],
    "vaca":                       ["cow"],
    "cavalo":                     ["horse"],
    "ovelha":                     ["sheep"],
    "macaco":                     ["monkey"],
    "bicicleta":                  ["bicycle"],
    "moto":                       ["motorcycle"],
    "carro":                      ["car"],
    "�nibus":                     ["bus"],
    "trem":                       ["train"],
    "caminh�o":                   ["truck"],
    "avi�o":                      ["airplane"],
    "barco":                      ["boat"],
    "skate":                      ["skateboard"],
    "prancha de surf":            ["surfboard"],
    "t�nis":                      ["tennis racket"],
    "mesa de jantar":             ["dining table"],
    "mesa":                       ["table", "desk", "dining table"],
    "cadeira":                    ["chair"],
    "sof�":                       ["couch", "sofa"],
    "cama":                       ["bed"],
    "vaso de planta":             ["potted plant"],
    "banheiro":                   ["toilet"],
    "televis�o":                  ["tv", "tvmonitor"],
    "abajur":                     ["lamp"],
    "espelho":                    ["mirror"],
    "laptop":                     ["laptop"],
    "computador":                 ["computer", "desktop computer", "tv"],
    "teclado":                    ["keyboard"],
    "mouse":                      ["mouse"],
    "controle remoto":            ["remote"],
    "celular":                    ["cell phone"],
    "micro-ondas":                ["microwave"],
    "forno":                      ["oven"],
    "torradeira":                 ["toaster"],
    "geladeira":                  ["refrigerator"],
    "caixa de som":               ["speaker"],
    "c�mera":                     ["camera"],
    "garrafa":                    ["bottle"],
    "copo":                       ["cup"],
    "ta�a de vinho":              ["wine glass"],
    "ta�a":                       ["wine glass", "cup"],
    "prato":                      ["plate", "dish"],
    "tigela":                     ["bowl"],
    "garfo":                      ["fork"],
    "faca":                       ["knife"],
    "colher":                     ["spoon"],
    "panela":                     ["pan", "pot"],
    "frigideira":                 ["skillet", "frying pan"],
    "martelo":                    ["hammer"],
    "chave inglesa":              ["wrench"],
    "furadeira":                  ["drill"],
    "parafusadeira":              ["drill"],
    "serra":                      ["saw"],
    "ro�adeira":                  ["brush cutter"],
    "alicate":                    ["pliers"],
    "chave de fenda":             ["screwdriver"],
    "lanterna":                   ["flashlight"],
    "fita m�trica":               ["tape measure"],
    "mochila":                    ["backpack"],
    "bolsa":                      ["handbag", "purse", "bag"],
    "carteira":                   ["wallet"],
    "�culos":                     ["glasses", "eyeglasses"],
    "rel�gio":                    ["clock", "watch"],
    "chinelo":                    ["sandal", "flip-flop"],
    "sapato":                     ["shoe"],
    "sandu�che":                  ["sandwich"],
    "hamb�rguer":                 ["hamburger"],
    "banana":                     ["banana"],
    "ma��":                       ["apple"],
    "laranja":                    ["orange"],
    "bolo":                       ["cake"],
    "rosquinha":                  ["donut"],
    "pizza":                      ["pizza"],
    "cachorro-quente":            ["hot dog"],
    "escova de dentes":           ["toothbrush"],
    "secador de cabelo":          ["hair drier", "hair dryer"],
    "cotonete":                   ["cotton swab"],
    "sacola pl�stica":            ["plastic bag"],
    "livro":                      ["book"],
    "vaso":                       ["vase"],
    "bola":                       ["sports ball", "ball"],
    "bexiga":                     ["balloon"],
    "pipa":                       ["kite"],
    "luva":                       ["glove"],
    "skis":                       ["skis"],
    "snowboard":                  ["snowboard"],
    "tesoura":                    ["scissors"],
}

# DeepFace
DB_PATH = os.path.join(BaseDir,"UserSettings", "known_faces")
DEEPFACE_MODEL_NAME = 'VGG-Face'
DEEPFACE_DETECTOR_BACKEND = 'opencv'
DEEPFACE_DISTANCE_METRIC = 'cosine'

# MiDaS
MIDAS_MODEL_TYPE = "MiDaS_small"
METERS_PER_STEP = 0.7

# --- Configura��o do Cliente Gemini ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    try:
        from dotenv import load_dotenv
        # Especifica o caminho para o .env se ele n�o estiver na raiz do projeto
        dotenv_path = os.path.join(BaseDir, '.env') # Ajuste se necess�rio
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            API_KEY = os.environ.get("GEMINI_API_KEY")
            logger.info(f"Chave API carregada de: {dotenv_path}")
        else:
             logger.info(f"Arquivo .env n�o encontrado em: {dotenv_path}")
    except ImportError:
        logger.info("Biblioteca python-dotenv n�o instalada. N�o � poss�vel carregar .env.")
        pass
    except Exception as e_env:
        logger.error(f"Erro ao carregar .env: {e_env}")


if not API_KEY:
    logger.info("AVISO: Chave da API Gemini n�o encontrada nas vari�veis de ambiente ou .env.")
    # Removido o fallback para chave placeholder para maior seguran�a.
    # O c�digo abaixo com a chave hardcoded ser� usado, mas N�O � RECOMENDADO.
    # API_KEY = "SUA_API_KEY_AQUI" # Substitua se necess�rio, mas prefira vari�veis de ambiente

# ATEN��O: A chave hardcoded abaixo ainda est� presente. Remova-a ou substitua por API_KEY.
# � ALTAMENTE RECOMENDADO usar a vari�vel API_KEY carregada acima.
try:
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"), # <-- SUBSTITUA por API_KEY ou remova se API_KEY for None
        # api_key=API_KEY, # <-- Use esta linha em vez da acima se API_KEY for carregada
        http_options=types.HttpOptions(api_version='v1alpha')
    )
    logger.info("Cliente Gemini inicializado.")
except Exception as e_client:
    logger.error(f"ERRO CR�TICO ao inicializar cliente Gemini: {e_client}")
    logger.info("Verifique a API Key e a conex�o.")
    client = None # Define como None para indicar falha
    # exit(1) # Descomente para sair se o cliente for essencial

# --- Ferramentas Gemini (Function Calling) ---
tools = [
    types.Tool(code_execution=types.ToolCodeExecution),
    types.Tool(google_search=types.GoogleSearch()),
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="save_known_face",
                description="Salva o rosto da pessoa atualmente em foco pela c�mera. Se 'person_name' n�o for fornecido, a IA deve solicitar o nome ao usu�rio com uma mensagem clara, como 'Por favor, informe o nome da pessoa para salvar o rosto.' Ap�s receber o nome, a fun��o salva o rosto e confirma o salvamento com 'Rosto salvo com sucesso para [nome].' Se a captura falhar, retorna 'Erro: N�o foi poss�vel capturar o rosto. Tente novamente.'",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "person_name": types.Schema(
                            type=types.Type.STRING,
                            description="Nome da pessoa a ser salvo. Se omitido, a IA deve solicitar ao usu�rio."
                        )
                    }
                )
            ),
            types.FunctionDeclaration(
                name="identify_person_in_front",
                description="Identifica a pessoa atualmente em foco pela c�mera usando o banco de dados de rostos conhecidos. Deve ser chamado apenas quando o usu�rio expressa explicitamente a inten��o de identificar uma pessoa. Se m�ltiplos rostos forem detectados, retorna o mais pr�ximo. Inclui a confian�a da identifica��o (ex: 'Identificado como [nome] com 95% de confian�a.'). Se n�o houver correspond�ncia, retorna 'Pessoa n�o reconhecida.'",
                parameters=types.Schema(type=types.Type.OBJECT, properties={})
            ),
            types.FunctionDeclaration(
                name="find_object_and_estimate_distance",
                description="Localiza um objeto espec�fico na vis�o da c�mera usando detec��o de objetos (YOLO) e estima sua dist�ncia em passos com MiDaS. O 'object_type' deve ser uma das categorias do modelo YOLO (ex: 'person', 'car', 'bottle'). Retorna a dire��o (frente, esquerda, direita), se est� sobre uma superf�cie (ex: mesa), e a dist�ncia estimada. Se o objeto n�o for encontrado, retorna 'Objeto n�o encontrado na cena.'",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "object_description": types.Schema(
                            type=types.Type.STRING,
                            description="Descri��o completa do objeto (ex: 'garrafa azul')."
                        ),
                        "object_type": types.Schema(
                            type=types.Type.STRING,
                            description="Tipo principal do objeto (ex: 'bottle'). Deve ser uma categoria v�lida do modelo YOLO."
                        )
                    },
                    required=["object_description", "object_type"]
                )
            )
        ]
    ),
]

# --- Carregar Instru��o do Sistema do Arquivo ---
system_instruction_text = "Voc� � um assistente prestativo." # Prompt padr�o m�nimo
try:
    if not os.path.exists(SYSTEM_INSTRUCTION_PATH):
         logger.warning(f"AVISO: Arquivo de instru��o do sistema n�o encontrado em '{SYSTEM_INSTRUCTION_PATH}'. Usando prompt padr�o.")
         exit()
    else:
        with open(SYSTEM_INSTRUCTION_PATH, 'r', encoding='utf-8') as f:
            system_instruction_text = f.read()
        logger.info(f"Instru��o do sistema carregada de: {SYSTEM_INSTRUCTION_PATH}")
except Exception as e:
    logger.error(f"Erro ao ler o arquivo de instru��o do sistema: {e}")
    logger.info("Usando um prompt padr�o m�nimo.")
    traceback.print_exc()


# --- Configura��o da Sess�o LiveConnect Gemini ---
CONFIG = types.LiveConnectConfig(
    temperature=0.2,
    response_modalities=["audio"], # AJUSTADO: Corrigido para ser uma lista de strings
    speech_config=types.SpeechConfig(
        language_code="pt-BR",
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Orus")
        )
    ),
    tools=tools,
    system_instruction=types.Content(
        parts=[
            types.Part.from_text(text=f"O nome do seu usu�rio � {trckuser}, "), # Usa f-string para formatar
            types.Part.from_text(text=system_instruction_text)
        ],
        role="system"
    )
)



# --- Classe Principal do Assistente ---
class AudioLoop:
    """
    Gerencia o loop principal do assistente multimodal.
    """
    def __init__(self, video_mode: str = DEFAULT_MODE, show_preview: bool = False):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            self.trckuser = cfg.get("trckuser", "Usu�rio")
        except Exception as e:
            logger.warning(f"N�o foi poss�vel ler {CONFIG_PATH}: {e}")
            self.trckuser = "Usu�rio"        
        self.video_mode = video_mode
        self.show_preview = show_preview if video_mode == "camera" else False
        self.audio_in_queue: Optional[asyncio.Queue] = None
        self.out_queue: Optional[asyncio.Queue] = None # Ser� recriado na conex�o
        self.cmd_queue: Optional[asyncio.Queue] = asyncio.Queue(maxsize=50) # Mantido, mas n�o usado explicitamente no fluxo principal

        self.thinking_event = asyncio.Event()
        self.session: Optional[genai.live.AsyncLiveSession] = None
        self.yolo_model: Optional[YOLO] = None
        self.preview_window_active: bool = False
        self.stop_event = asyncio.Event()
        self.frame_lock = threading.Lock()
        self.latest_bgr_frame: Optional[np.ndarray] = None
        self.latest_yolo_results: Optional[List[Any]] = None

        # --- Novo estado para o fluxo de salvar rosto ---
        self.awaiting_name_for_save_face: bool = False

        # --- Carregamento de Modelos ---
        if self.video_mode == "camera":
            try:
                self.yolo_model = YOLO(YOLO_MODEL_PATH)
                logger.info(f"Modelo YOLO '{YOLO_MODEL_PATH}' carregado.")
            except FileNotFoundError:
                 logger.error(f"ERRO: Modelo YOLO n�o encontrado em '{YOLO_MODEL_PATH}'. YOLO desabilitado.")
                 self.yolo_model = None
            except Exception as e:
                logger.error(f"Erro ao carregar o modelo YOLO: {e}. YOLO desabilitado.")
                traceback.print_exc()
                self.yolo_model = None

        if not os.path.exists(DB_PATH):
            try:
                os.makedirs(DB_PATH)
                logger.info(f"Diret�rio DeepFace DB criado em: {DB_PATH}")
            except Exception as e:
                logger.error(f"Erro ao criar diret�rio {DB_PATH}: {e}")

        try:
            logger.info("Pr�-carregando modelos DeepFace...")
            # Cria um frame dummy para for�ar o carregamento
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            # Executa uma a��o leve (emo��o) sem exigir detec��o
            DeepFace.analyze(img_path=dummy_frame, actions=['emotion'], enforce_detection=False)
            logger.info("Modelos DeepFace pr�-carregados.")
        except Exception as e:
            logger.warning(f"AVISO: Erro ao pr�-carregar modelos DeepFace: {e}.")
            # traceback.print_exc() # Descomente se precisar depurar o pr�-carregamento

        self.midas_model = None
        self.midas_transform = None
        self.midas_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            logger.info(f"Carregando modelo MiDaS ({MIDAS_MODEL_TYPE}) para {self.midas_device}...")
            # Carrega o modelo MiDaS do Torch Hub
            self.midas_model = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE)
            # Carrega as transforma��es correspondentes
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if MIDAS_MODEL_TYPE == "MiDaS_small":
                 self.midas_transform = midas_transforms.small_transform
            else:
                 # Assume DPT transform para outros modelos (pode precisar ajustar)
                 self.midas_transform = midas_transforms.dpt_transform
            # Move o modelo para o dispositivo (GPU ou CPU)
            self.midas_model.to(self.midas_device)
            # Define o modelo para modo de avalia��o (desativa dropout, etc.)
            self.midas_model.eval()
            logger.info("Modelo MiDaS carregado.")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo MiDaS: {e}. Estimativa de profundidade desabilitada.")
            # traceback.print_exc() # Descomente para depurar o carregamento do MiDaS
            self.midas_model = None
            self.midas_transform = None
          
 


    async def send_text(self):
        """L� input de texto do usu�rio, trata comandos de debug e envia ao Gemini."""
        logger.info("Pronto para receber comandos de texto. Digite 'q' para sair, 'p' para salvar rosto (debug).")
        while not self.stop_event.is_set():
            try:
                # L� input do usu�rio em uma thread separada para n�o bloquear asyncio
                text = await asyncio.to_thread(input, "message > ")
                if self.out_queue:
                    while not self.out_queue.empty():
                        self.out_queue.get_nowait()
                        self.out_queue.task_done()
                        

                # --- Tratamento de Comandos Locais/Debug ---   
                if text.lower() == "q":
                    self.stop_event.set()
                    logger.info("Sinal de parada ('q') recebido. Encerrando...")
                    break # Sai do loop while

                # --- ADI��O: Comando de Debug 'p' ---
                elif text.lower() == "p":
                    logger.info("[DEBUG] Comando 'p' recebido. Tentando salvar rosto como 'pedro'...")
                    if self.video_mode == "camera":
                        # Chama a fun��o de salvar rosto diretamente em outra thread
                        try:
                            # Bloqueia o 'pensamento' para evitar conflitos? Opcional.
                            # self.thinking_event.set()
                            logger.info("  [DEBUG] Chamando _handle_save_known_face('pedro')...")
                            result = await asyncio.to_thread(self._handle_save_known_face, "pedro")
                            logger.info(f"  [DEBUG] Resultado do salvamento direto: {result}")
                        except Exception as e_debug_save:
                            logger.info(f"  [DEBUG] Erro ao tentar salvar rosto diretamente: {e_debug_save}")
                            traceback.print_exc()
                        # finally:
                            # self.thinking_event.clear()
                    else:
                        logger.info("  [DEBUG] Salvar rosto s� funciona no modo c�mera.")
                    continue # Pula o envio para Gemini e espera o pr�ximo input

                # --- Envio Normal para Gemini ---
                # Verifica se a sess�o existe e est� ativa antes de enviar
                if self.session:
                    logger.info(f"Enviando texto para Gemini: '{text}'")
                    # Envia o texto para o Gemini, marcando o fim do turno do usu�rio
                    # Envia "." se o texto estiver vazio para manter a sess�o ativa
                    await self.session.send(input=text or ".", end_of_turn=True)
                else:
                    # Avisa se a sess�o n�o estiver ativa
                    if not self.stop_event.is_set():
                        logger.info("Sess�o Gemini n�o est� ativa. N�o � poss�vel enviar mensagem.")
                        await asyncio.sleep(0.5) # Evita spamming da mensagem
                        

            except asyncio.CancelledError:
                logger.info("send_text cancelado.")
                break
            except Exception as e:
                logger.error(f"Erro em send_text: {e}")
                # Adiciona traceback para depura��o em caso de erro inesperado
                # traceback.print_exc()
                # Verifica se o erro indica sess�o fechada para parar
                error_str_upper = str(e).upper()
                if "LIVESESSION CLOSED" in error_str_upper or "LIVESESSION NOT CONNECTED" in error_str_upper:
                    logger.info("Erro em send_text indica sess�o fechada. Sinalizando parada.")
                    self.stop_event.set()
                # Considerar parar em outros erros graves tamb�m?
                # else: self.stop_event.set()
                break # Sai do loop em caso de erro
        logger.info("send_text finalizado.")


    def _get_frame(self, cap: cv2.VideoCapture) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        # (Fun��o _get_frame inalterada - omitida para brevidade)
        ret, frame = cap.read()
        latest_frame_copy = None
        current_yolo_results = None

        if ret:
            latest_frame_copy = frame.copy()

        yolo_alerts = []
        display_frame = None
        if ret and self.yolo_model:
            frame_rgb = cv2.cvtColor(latest_frame_copy, cv2.COLOR_BGR2RGB)
            try:
                results = self.yolo_model.predict(frame_rgb, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
                current_yolo_results = results

                if self.show_preview:
                    display_frame = latest_frame_copy.copy()

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        class_name_yolo = self.yolo_model.names[cls_id] # Renomeado para evitar conflito
                        conf = float(box.conf[0])

                        if display_frame is not None:
                            label = f"{class_name_yolo} {conf:.2f}"
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        # Verifica se class_name_yolo est� em alguma das listas dentro de DANGER_CLASSES.values()
                        is_dangerous = any(class_name_yolo in danger_list for danger_list in DANGER_CLASSES.values())
                        if is_dangerous and conf >= YOLO_CONFIDENCE_THRESHOLD:
                            yolo_alerts.append(class_name_yolo)
            except Exception as e:
                logger.error(f"Erro na infer�ncia YOLO: {e}")
                # traceback.print_exc() # Descomente para depura��o detalhada do YOLO
                current_yolo_results = None
        elif self.show_preview and ret:
            display_frame = latest_frame_copy.copy()

        with self.frame_lock:
            if ret:
                self.latest_bgr_frame = latest_frame_copy
                self.latest_yolo_results = current_yolo_results
            else:
                self.latest_bgr_frame = None
                self.latest_yolo_results = None
                return None, []

        if self.show_preview and display_frame is not None:
            try:
                cv2.imshow("Trackie YOLO Preview", display_frame)
                cv2.waitKey(1)
                self.preview_window_active = True
            except cv2.error as e:
                if "DISPLAY" in str(e).upper() or "GTK" in str(e).upper() or "QT" in str(e).upper() or "COULD NOT CONNECT TO DISPLAY" in str(e).upper() or "plugin \"xcb\"" in str(e):
                    logger.info("--------------------------------------------------------------------")
                    logger.info("AVISO: N�o foi poss�vel mostrar a janela de preview da c�mera.")
                    logger.info("Desabilitando feedback visual para esta sess�o.")
                    logger.info("--------------------------------------------------------------------")
                    self.show_preview = False
                    self.preview_window_active = False
                else:
                    logger.error(f"Erro inesperado no OpenCV ao tentar mostrar preview: {e}")
                    # traceback.print_exc() # Descomente para depura��o detalhada do OpenCV
            except Exception as e_gen:
                logger.error(f"Erro geral ao tentar mostrar preview: {e_gen}")
                # traceback.print_exc() # Descomente para depura��o detalhada
                self.show_preview = False
                self.preview_window_active = False

        image_part = None
        if ret:
            try:
                if 'frame_rgb' not in locals() or frame_rgb is None: # Adicionado 'or frame_rgb is None'
                     frame_rgb = cv2.cvtColor(latest_frame_copy, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img.thumbnail([1024, 1024]) # Redimensiona mantendo propor��o
                image_io = io.BytesIO()
                img.save(image_io, format="jpeg", quality=85) # Ajusta qualidade JPEG
                image_io.seek(0)
                image_part = {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_io.read()).decode('utf-8')
                }
            except Exception as e:
                logger.error(f"Erro na convers�o do frame para JPEG: {e}")
                # traceback.print_exc() # Descomente para depura��o detalhada

        return image_part, list(set(yolo_alerts))


    async def get_frames(self):
        # (Fun��o get_frames inalterada, exceto por traceback em erro cr�tico - omitida para brevidade)
        cap = None
        try:
            logger.info("Iniciando captura da c�mera...")
            cap = await asyncio.to_thread(cv2.VideoCapture, 0)
            target_fps = 1
            # Tenta definir FPS, mas n�o � garantido que funcione em todas as c�meras/drivers
            cap.set(cv2.CAP_PROP_FPS, target_fps)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"FPS solicitado: {target_fps}, FPS real da c�mera: {actual_fps if actual_fps > 0 else 'N�o dispon�vel'}")

            if actual_fps > 0 and actual_fps < target_fps * 5: # Usa FPS real se razo�vel
                sleep_interval = 1 / actual_fps
            else:
                sleep_interval = 1 / target_fps # Usa FPS alvo como fallback
            sleep_interval = max(0.1, min(sleep_interval, 1.0)) # Limita entre 0.1s e 1.0s
            logger.info(f"Intervalo de captura de frame: {sleep_interval:.2f}s")


            if not cap.isOpened():
                logger.info("Erro: N�o foi poss�vel abrir a c�mera.")
                with self.frame_lock:
                    self.latest_bgr_frame = None
                    self.latest_yolo_results = None
                self.stop_event.set()
                return

            while not self.stop_event.is_set():
                if not cap.isOpened():
                    logger.info("C�mera desconectada ou fechada inesperadamente.")
                    self.stop_event.set()
                    break

                # Executa a captura e processamento s�ncrono em outra thread
                image_part, yolo_alerts = await asyncio.to_thread(self._get_frame, cap)

                with self.frame_lock:
                    frame_was_read = self.latest_bgr_frame is not None

                if not frame_was_read:
                     if not cap.isOpened():
                         logger.info("Leitura do frame falhou e c�mera fechada. Encerrando get_frames.")
                         self.stop_event.set()
                         break
                     else:
                         logger.info("Aviso: Falha tempor�ria na leitura do frame.")
                         await asyncio.sleep(0.5) # Espera um pouco antes de tentar de novo
                         continue

                # Envia frame para a fila de sa�da
                if image_part is not None and self.out_queue:
                    try:
                        if self.out_queue.full():
                            # Descarta o mais antigo para dar espa�o ao novo
                            discarded = await self.out_queue.get()
                            self.out_queue.task_done() # Marca o descartado como conclu�do
                            # logger.info("Aviso: Fila de sa�da cheia, descartando frame antigo.") # Log opcional
                        self.out_queue.put_nowait(image_part)
                    except asyncio.QueueFull:
                         # Isso n�o deveria acontecer se a l�gica acima estiver correta, mas por seguran�a
                         pass # logger.info("Aviso: Fila de sa�da ainda cheia ao tentar enfileirar frame.") # Log opcional
                    except Exception as q_e:
                         logger.error(f"Erro inesperado ao manipular out_queue em get_frames: {q_e}")


                # Envia alertas YOLO urgentes diretamente
                if yolo_alerts and self.session:
                    for alert_class_name in yolo_alerts:
                        try:
                            alert_msg = f"YOLO DIZENDO (DETEC��O DE PERIGOS):TRACKIE CUIDADO! AVISE AO {self.trckuser} QUE {alert_class_name.upper()} FOI DETECTADO!"
                           # play_wav_file_sync(DANGER_SOUND_PATH)
                            # N�o precisa verificar out_queue aqui, pois envia direto
                            await self.session.send(input=alert_msg, end_of_turn=True)
                            logger.info(f"ALERTA URGENTE ENVIADO: {alert_msg}")
                        except Exception as e:
                            logger.error(f"Erro ao enviar alerta urgente: {e}")
                            if "LiveSession closed" in str(e) or "LiveSession not connected" in str(e):
                                logger.info("Erro ao enviar alerta indica sess�o fechada. Sinalizando parada.")
                                self.stop_event.set()
                                break # Sai do loop de alertas se a sess�o fechar

                # Aguarda antes do pr�ximo ciclo
                await asyncio.sleep(sleep_interval)

        except asyncio.CancelledError:
            logger.info("get_frames cancelado.")
        except Exception as e:
            logger.error(f"Erro cr�tico em get_frames: {e}")
            traceback.print_exc() # Imprime traceback para erros cr�ticos
            self.stop_event.set()
        finally:
            logger.info("Finalizando get_frames...")
            if cap and cap.isOpened():
                cap.release()
                logger.info("C�mera liberada.")
            # Garante que o estado do frame seja limpo
            with self.frame_lock:
                self.latest_bgr_frame = None
                self.latest_yolo_results = None
            # Garante que a janela de preview seja fechada
            if self.preview_window_active:
                try:
                    # Tenta fechar a janela espec�fica primeiro
                    cv2.destroyWindow("Trackie YOLO Preview")
                    logger.info("Janela de preview 'Trackie YOLO Preview' fechada.")
                except Exception:
                    try:
                        # Se falhar, tenta fechar todas as janelas OpenCV
                        cv2.destroyAllWindows()
                        logger.info("Todas as janelas OpenCV fechadas.")
                    except Exception as e_cv_destroy_all:
                        logger.warning(f"AVISO: erro ao tentar fechar janelas de preview no finally: {e_cv_destroy_all}")
            self.preview_window_active = False # Garante que o estado est� correto
            logger.info("get_frames conclu�do.")


    def _get_screen(self) -> Optional[Dict[str, Any]]:
        # (Fun��o _get_screen inalterada - omitida para brevidade)
        sct = mss.mss()
        monitor_number = 1 # Tenta usar o monitor 1 (geralmente o principal)
        try:
            monitors = sct.monitors
            if len(monitors) > monitor_number:
                 monitor = monitors[monitor_number]
            elif monitors: # Se n�o houver monitor 1, mas houver algum monitor (geralmente o monitor 0 � 'todos')
                 monitor = monitors[0] # Usa o primeiro monitor dispon�vel (pode ser 'todos')
                 if len(monitors) > 1: # Se houver mais de um, pega o segundo (�ndice 1), que geralmente � o prim�rio real
                     monitor = monitors[1]
            else: # Nenhum monitor encontrado
                logger.info("Erro: Nenhum monitor detectado por mss.")
                return None

            # Captura a imagem do monitor selecionado
            sct_img = sct.grab(monitor)

            # Cria a imagem PIL a partir dos dados brutos BGR
            img = Image.frombytes('RGB', sct_img.size, sct_img.rgb, 'raw', 'BGR')
            # N�o precisa converter para RGB se salvar como PNG, mas n�o faz mal
            # img = img.convert('RGB')

            # Salva em mem�ria como PNG (melhor para capturas de tela que JPEG)
            image_io = io.BytesIO()
            img.save(image_io, format="PNG")
            image_io.seek(0)

            # Codifica em Base64 e retorna no formato esperado
            return {
                "mime_type": "image/png",
                "data": base64.b64encode(image_io.read()).decode('utf-8')
            }
        except Exception as e:
            logger.error(f"Erro ao capturar tela: {e}")
            # traceback.print_exc() # Descomente para depura��o detalhada do mss
            return None
   
   
    async def get_screen(self):
        # (Fun��o get_screen inalterada, exceto por traceback em erro cr�tico - omitida para brevidade)
        logger.info("Iniciando captura de tela...")
        try:
            while not self.stop_event.is_set():
                # Executa a captura s�ncrona em outra thread
                frame_data = await asyncio.to_thread(self._get_screen)

                if frame_data is None:
                    logger.info("Falha ao capturar frame da tela.")
                    await asyncio.sleep(1.0) # Espera antes de tentar novamente
                    continue

                # Envia frame para a fila de sa�da
                if self.out_queue:
                    try:
                         if self.out_queue.full():
                             # Descarta o mais antigo
                             discarded = await self.out_queue.get()
                             self.out_queue.task_done()
                             # logger.info("Aviso: Fila de sa�da cheia, descartando frame de tela antigo.") # Log opcional
                         self.out_queue.put_nowait(frame_data)
                    except asyncio.QueueFull:
                         pass # logger.info("Aviso: Fila de sa�da ainda cheia ao tentar enfileirar frame de tela.") # Log opcional
                    except Exception as q_e:
                         logger.error(f"Erro inesperado ao manipular out_queue em get_screen: {q_e}")

                # Espera 1 segundo entre capturas de tela
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("get_screen cancelado.")
        except Exception as e:
            logger.error(f"Erro cr�tico em get_screen: {e}")
            traceback.print_exc() # Imprime traceback para erros cr�ticos
            self.stop_event.set()
        finally:
            logger.info("get_screen finalizado.")


    async def send_realtime(self):
        # (Fun��o send_realtime inalterada, exceto por traceback em erro cr�tico - omitida para brevidade)
        logger.info("Send_realtime pronto para enviar dados...")
        try:
            while not self.stop_event.is_set():
                # Pausa se o Gemini estiver processando uma fun��o (thinking_event)
                if self.thinking_event.is_set():
                    await asyncio.sleep(0.05)
                    continue

                # Verifica se a fila de sa�da existe (pode ser None durante reconex�o)
                if not self.out_queue:
                    await asyncio.sleep(0.1)
                    continue

                # Tenta obter uma mensagem da fila com timeout
                try:
                    msg = await asyncio.wait_for(self.out_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Timeout � normal se n�o houver dados por 1s
                    continue
                except asyncio.QueueEmpty:
                    # Fila vazia tamb�m � normal
                    continue
                except Exception as q_get_e:
                    # Erro inesperado ao obter da fila
                    logger.error(f"Erro ao obter da out_queue em send_realtime: {q_get_e}")
                    await asyncio.sleep(0.1)
                    continue


                # Verifica se a sess�o Gemini est� ativa
                if not self.session:
                    # logger.info("Sess�o Gemini n�o est� ativa (send_realtime). Descartando mensagem.") # Log opcional
                    if self.out_queue: self.out_queue.task_done() # Marca a tarefa como conclu�da mesmo descartando
                    if not self.stop_event.is_set():
                        await asyncio.sleep(0.5)
                    continue

                # Tenta enviar a mensagem para o Gemini
                try:
                    if isinstance(msg, dict) and "data" in msg and "mime_type" in msg : # Mensagem multimodal (imagem/�udio)
                        await self.session.send(input=msg, end_of_turn=True)
                    elif isinstance(msg, str): # Mensagem de texto (raro neste fluxo)
                        # AJUSTADO: end_of_turn=True para garantir que o modelo processe como um turno completo
                        # se uma string chegar aqui (ex: alerta sist�mico ou, hipoteticamente, msg de usu�rio perdida).
                        # Inputs de usu�rio normais s�o via send_text (com end_of_turn=True).
                        # Alertas YOLO s�o enviados de get_frames (com end_of_turn=True).
                        logger.info(f"Enviando texto via send_realtime (tratando como turno completo): {msg}")
                        await self.session.send(input=msg, end_of_turn=True)
                    else:
                        logger.info(f"Mensagem desconhecida em send_realtime: {type(msg)}")

                    # Marca a tarefa como conclu�da na fila ap�s envio bem-sucedido
                    if self.out_queue: self.out_queue.task_done()

                except Exception as e_send:
                    logger.error(f"Erro ao enviar para Gemini em send_realtime: {e_send}")
                    # Garante task_done mesmo em erro para n�o bloquear a fila
                    if self.out_queue: self.out_queue.task_done()
                    # Verifica se o erro indica sess�o fechada/perdida
                    error_str_upper = str(e_send).upper()
                    if "LIVESESSION CLOSED" in error_str_upper or \
                       "LIVESESSION NOT CONNECTED" in error_str_upper or \
                       "DEADLINE EXCEEDED" in error_str_upper or \
                       "RST_STREAM" in error_str_upper or \
                       "UNAVAILABLE" in error_str_upper:
                        logger.info("Erro de envio indica sess�o Gemini fechada/perdida. Sinalizando parada.")
                        self.stop_event.set()
                        break # Sai do loop while
                    else:
                        # Outros erros podem ser tempor�rios, imprime traceback para an�lise
                        traceback.print_exc()
                        await asyncio.sleep(0.5) # Pausa antes de tentar processar pr�ximo item

        except asyncio.CancelledError:
            logger.info("send_realtime cancelado.")
        except Exception as e:
            logger.error(f"Erro fatal em send_realtime: {e}")
            traceback.print_exc() # Imprime traceback para erros fatais
            self.stop_event.set()
        finally:
            logger.info("send_realtime finalizado.")


    async def listen_audio(self):
        # (Fun��o listen_audio inalterada, exceto por traceback em erro cr�tico - omitida para brevidade)
        if not pya:
            logger.info("PyAudio n�o inicializado. Tarefa listen_audio n�o pode iniciar.")
            return

        audio_stream = None
        try:
            logger.info("Configurando stream de �udio de entrada...")
            mic_info = pya.get_default_input_device_info()
            logger.info(f"Usando microfone: {mic_info['name']}")
            audio_stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE,
                input=True, input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE
            )
            logger.info("Escutando �udio do microfone...")

            while not self.stop_event.is_set():
                # Pausa se o Gemini estiver processando
                if self.thinking_event.is_set():
                    await asyncio.sleep(0.05)
                    continue

                # Verifica se o stream ainda est� ativo
                if not audio_stream or not audio_stream.is_active():
                     logger.info("Stream de �udio de entrada n�o est� ativo. Encerrando listen_audio.")
                     self.stop_event.set()
                     break

                # L� dados do microfone em outra thread
                try:
                    data = await asyncio.to_thread(
                        audio_stream.read, CHUNK_SIZE, exception_on_overflow=False
                    )
                    # Envia dados para a fila de sa�da
                    if self.out_queue:
                         try:
                             if self.out_queue.full():
                                 # Descarta o mais antigo se a fila estiver cheia (menos prov�vel para �udio)
                                 discarded = await self.out_queue.get()
                                 self.out_queue.task_done()
                                 # pass # Ou simplesmente n�o envia se estiver cheio
                                 # logger.info("Aviso: Fila de sa�da cheia, �udio pode ser atrasado/descartado.")
                             self.out_queue.put_nowait({"data": data, "mime_type": "audio/pcm"})
                         except asyncio.QueueFull:
                             pass # logger.info("Aviso: Fila de sa�da cheia ao tentar enfileirar �udio.")
                         except Exception as q_e:
                              logger.error(f"Erro inesperado ao manipular out_queue em listen_audio: {q_e}")

                except OSError as e_os:
                    # Erros comuns de stream fechado ou overflow
                    if e_os.errno == -9988 or "Stream closed" in str(e_os) or "Input overflowed" in str(e_os):
                        logger.info(f"Stream de �udio fechado ou com overflow (OSError: {e_os}). Encerrando listen_audio.")
                        self.stop_event.set()
                        break
                    else:
                        # Outros erros de OS podem ser mais s�rios
                        logger.error(f"Erro de OS ao ler do stream de �udio: {e_os}")
                        traceback.print_exc()
                        self.stop_event.set()
                        break
                except Exception as e_read:
                    # Erro gen�rico durante a leitura
                    logger.error(f"Erro durante a leitura do �udio em listen_audio: {e_read}")
                    traceback.print_exc()
                    self.stop_event.set() # Para a tarefa em caso de erro de leitura
                    break
        except asyncio.CancelledError:
            logger.info("listen_audio cancelado.")
        except Exception as e:
            # Erro cr�tico na configura��o ou loop principal
            logger.error(f"Erro cr�tico em listen_audio: {e}")
            traceback.print_exc()
            self.stop_event.set()
        finally:
            logger.info("Finalizando listen_audio...")
            # Garante que o stream seja fechado
            if audio_stream:
                try:
                    if audio_stream.is_active():
                        audio_stream.stop_stream()
                    audio_stream.close()
                    logger.info("Stream de �udio de entrada fechado.")
                except Exception as e_close_stream:
                    logger.error(f"Erro ao fechar stream de �udio de entrada: {e_close_stream}")
            logger.info("listen_audio conclu�do.")


    def _handle_save_known_face(self, person_name: str) -> str:
        """Processa a chamada de fun��o para salvar um rosto."""
        # --- LOGGING IN�CIO ---
        logger.info("[LOG] Executando: _handle_save_known_face")
        logger.info(f"[LOG]   - Argumentos: person_name='{person_name}'")
        # --- FIM LOGGING ---

        start_time = time.time()
        result_message = "" # Inicializa a mensagem de resultado

        logger.info(f"[DeepFace] Iniciando salvamento para: {person_name}")
        frame_to_process = None
        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy()

        if frame_to_process is None:
            logger.info("[DeepFace] Erro: Nenhum frame dispon�vel para salvar.")
            result_message = "N�o foi poss�vel capturar a imagem para salvar o rosto."
            # --- LOGGING FIM ---
            logger.info("[LOG] Finalizado: _handle_save_known_face (Erro: Sem frame)")
            logger.info(f"[LOG]   - Dura��o: {time.time() - start_time:.2f}s")
            logger.info(f"[LOG]   - Resultado: '{result_message}'")
            # --- FIM LOGGING ---
            return result_message

        # Sanitiza nome para diret�rio e arquivo
        safe_person_name_dir = "".join(c if c.isalnum() or c in [' '] else '_' for c in person_name).strip().replace(" ", "_")
        if not safe_person_name_dir: safe_person_name_dir = "desconhecido"
        person_dir = os.path.join(DB_PATH, safe_person_name_dir)

        try:
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
                logger.info(f"[DeepFace] Diret�rio criado: {person_dir}")

            # Extrai rosto(s) do frame
            detected_faces = DeepFace.extract_faces(
                img_path=frame_to_process,
                detector_backend=DEEPFACE_DETECTOR_BACKEND,
                enforce_detection=True, # Garante que um rosto foi detectado
                align=True,
               # silent=True
            )

            # Verifica se algum rosto foi detectado
            if not detected_faces or not isinstance(detected_faces, list) or 'facial_area' not in detected_faces[0]:
                logger.info(f"[DeepFace] Nenhum rosto detectado para {person_name}.")
                result_message = f"N�o consegui detectar um rosto claro para {person_name}."
                # --- LOGGING FIM ---
                logger.info("[LOG] Finalizado: _handle_save_known_face (Erro: Rosto n�o detectado)")
                logger.info(f"[LOG]   - Dura��o: {time.time() - start_time:.2f}s")
                logger.info(f"[LOG]   - Resultado: '{result_message}'")
                # --- FIM LOGGING ---
                return result_message

            # Pega o primeiro rosto detectado (geralmente o maior/mais central)
            face_data = detected_faces[0]['facial_area']
            x, y, w, h = face_data['x'], face_data['y'], face_data['w'], face_data['h']

            # Recorta a imagem do rosto com uma margem
            margin = 10
            y1, y2 = max(0, y - margin), min(frame_to_process.shape[0], y + h + margin)
            x1, x2 = max(0, x - margin), min(frame_to_process.shape[1], x + w + margin)
            face_image = frame_to_process[y1:y2, x1:x2]

            if face_image.size == 0:
                 logger.info(f"[DeepFace] Erro ao recortar rosto para {person_name} (imagem vazia).")
                 result_message = f"Erro ao processar o rosto de {person_name}."
                 # --- LOGGING FIM ---
                 logger.info("[LOG] Finalizado: _handle_save_known_face (Erro: Recorte vazio)")
                 logger.info(f"[LOG]   - Dura��o: {time.time() - start_time:.2f}s")
                 logger.info(f"[LOG]   - Resultado: '{result_message}'")
                 # --- FIM LOGGING ---
                 return result_message

            # Cria nome de arquivo �nico
            timestamp = int(time.time())
            safe_file_name_base = "".join(c if c.isalnum() else '_' for c in person_name).strip()
            if not safe_file_name_base: safe_file_name_base = "rosto"
            file_name = f"{safe_file_name_base.lower()}_{timestamp}.jpg"
            file_path = os.path.join(person_dir, file_name)

            # Salva a imagem do rosto
            save_success = cv2.imwrite(file_path, face_image)
            if not save_success:
                logger.info(f"[DeepFace] Erro ao salvar imagem em {file_path}")
                result_message = f"Erro ao salvar a imagem do rosto de {person_name}."
                # --- LOGGING FIM ---
                logger.info("[LOG] Finalizado: _handle_save_known_face (Erro: Falha no imwrite)")
                logger.info(f"[LOG]   - Dura��o: {time.time() - start_time:.2f}s")
                logger.info(f"[LOG]   - Resultado: '{result_message}'")
                # --- FIM LOGGING ---
                return result_message


            # Remove cache de representa��es para for�ar rec�lculo
            model_name_safe = DEEPFACE_MODEL_NAME.lower().replace('-', '_')
            representations_pkl_path = os.path.join(DB_PATH, f"representations_{model_name_safe}.pkl")
            if os.path.exists(representations_pkl_path):
                try:
                    os.remove(representations_pkl_path)
                    logger.info(f"[DeepFace] Cache de representa��es '{representations_pkl_path}' removido para atualiza��o.")
                except Exception as e_pkl:
                    logger.info(f"[DeepFace] Aviso: Falha ao remover cache de representa��es: {e_pkl}")

            logger.info(f"[DeepFace] Rosto de {person_name} salvo em {file_path}")
            result_message = f"Rosto de {person_name} salvo com sucesso."

        except ValueError as ve: # Captura erro espec�fico do DeepFace se enforce_detection=True falhar
             logger.info(f"[DeepFace] Nenhum rosto detectado (ValueError) para {person_name}: {ve}")
             result_message = f"N�o consegui detectar um rosto claro para salvar para {person_name}."
        except Exception as e:
            logger.info(f"[DeepFace] Erro inesperado ao salvar rosto para {person_name}: {e}")
            traceback.print_exc()
            result_message = f"Ocorreu um erro ao tentar salvar o rosto de {person_name}."

        # --- LOGGING FIM ---
        logger.info("[LOG] Finalizado: _handle_save_known_face")
        logger.info(f"[LOG]   - Dura��o: {time.time() - start_time:.2f}s")
        logger.info(f"[LOG]   - Resultado: '{result_message}'")
        # --- FIM LOGGING ---
        return result_message


    def _handle_identify_person_in_front(self) -> str:
        """Processa a chamada de fun��o para identificar uma pessoa."""
        # --- LOGGING IN�CIO ---
        logger.info("[LOG] Executando: _handle_identify_person_in_front")
        # --- FIM LOGGING ---

        start_time = time.time()
        result_message = ""

        # Verifica se pandas foi importado com sucesso
        if pd is None:
            logger.info("[DeepFace] Erro: Biblioteca 'pandas' n�o est� dispon�vel. Identifica��o desabilitada.")
            result_message = "Erro interno: depend�ncia 'pandas' faltando para identifica��o."
            # --- LOGGING FIM ---
            logger.info("[LOG] Finalizado: _handle_identify_person_in_front (Erro: Sem pandas)")
            logger.info(f"[LOG]   - Dura��o: {time.time() - start_time:.2f}s")
            logger.info(f"[LOG]   - Resultado: '{result_message}'")
            # --- FIM LOGGING ---
            return result_message


        logger.info("[DeepFace] Iniciando identifica��o de pessoa...")
        frame_to_process = None
        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy()

        if frame_to_process is None:
            logger.info("[DeepFace] Erro: Nenhum frame dispon�vel para identificar.")
            result_message = "N�o foi poss�vel capturar a imagem para identificar."
            # --- LOGGING FIM ---
            logger.info("[LOG] Finalizado: _handle_identify_person_in_front (Erro: Sem frame)")
            logger.info(f"[LOG]   - Dura��o: {time.time() - start_time:.2f}s")
            logger.info(f"[LOG]   - Resultado: '{result_message}'")
            # --- FIM LOGGING ---
            return result_message

        try:
            # Usa DeepFace.find para buscar no banco de dados
            # enforce_detection=True: Exige detec��o clara de rosto na imagem de entrada
            # silent=True: Reduz o output do DeepFace
            dfs = DeepFace.find(
                img_path=frame_to_process,
                db_path=DB_PATH,
                model_name=DEEPFACE_MODEL_NAME,
                detector_backend=DEEPFACE_DETECTOR_BACKEND,
                distance_metric=DEEPFACE_DISTANCE_METRIC,
                enforce_detection=True, # Alterado para True para buscar rostos mais claros
                #silent=True,
                align=True
            )

            # DeepFace.find retorna uma lista de DataFrames. Pegamos o primeiro.
            # Verifica se a lista ou o DataFrame est�o vazios
            if not dfs or not isinstance(dfs, list) or not isinstance(dfs[0], pd.DataFrame) or dfs[0].empty:
                logger.info("[DeepFace] Nenhuma correspond�ncia encontrada ou rosto n�o detectado claramente.")
                result_message = "N�o consegui reconhecer ningu�m ou n�o detectei um rosto claro."
                # --- LOGGING FIM ---
                logger.info("[LOG] Finalizado: _handle_identify_person_in_front (Nenhuma correspond�ncia)")
                logger.info(f"[LOG]   - Dura��o: {time.time() - start_time:.2f}s")
                logger.info(f"[LOG]   - Resultado: '{result_message}'")
                # --- FIM LOGGING ---
                return result_message

            df = dfs[0]

            # Encontra a coluna de dist�ncia correta (pode variar com modelo/m�trica)
            distance_col_name = f"{DEEPFACE_MODEL_NAME}_{DEEPFACE_DISTANCE_METRIC}"
            if distance_col_name not in df.columns:
                # Fallback para nomes comuns ou que contenham a m�trica
                if 'distance' in df.columns:
                    distance_col_name = 'distance'
                else:
                    found_col = None
                    for col in df.columns:
                        if DEEPFACE_DISTANCE_METRIC in col.lower():
                            found_col = col
                            break
                    if found_col:
                        distance_col_name = found_col
                    else:
                        logger.info(f"[DeepFace] Erro: Coluna de dist�ncia '{distance_col_name}' ou similar n�o encontrada. Colunas: {df.columns.tolist()}")
                        result_message = "Erro ao processar resultado da identifica��o (coluna de dist�ncia)."
                        # --- LOGGING FIM ---
                        logger.info("[LOG] Finalizado: _handle_identify_person_in_front (Erro: Coluna dist�ncia)")
                        logger.info(f"[LOG]   - Dura��o: {time.time() - start_time:.2f}s")
                        logger.info(f"[LOG]   - Resultado: '{result_message}'")
                        # --- FIM LOGGING ---
                        return result_message

            # Ordena por dist�ncia (menor � melhor) e pega o melhor resultado
            df = df.sort_values(by=distance_col_name, ascending=True)
            best_match = df.iloc[0]

            # Extrai informa��es do melhor match
            best_match_identity_path = best_match['identity']
            # O nome da pessoa � o nome do diret�rio pai do arquivo de imagem
            person_name = os.path.basename(os.path.dirname(best_match_identity_path))
            distance = best_match[distance_col_name]

            logger.info(f"[DeepFace] Pessoa potencialmente identificada: {person_name} (Dist�ncia: {distance:.4f})")

            # Define limiares de dist�ncia (ajustar experimentalmente!)
            thresholds = {
                'VGG-Face': {'cosine': 0.40, 'euclidean': 0.60, 'euclidean_l2': 0.86},
                'Facenet': {'cosine': 0.40, 'euclidean': 0.90, 'euclidean_l2': 1.10},
                'Facenet512': {'cosine': 0.30, 'euclidean': 0.70, 'euclidean_l2': 0.95},
                'ArcFace': {'cosine': 0.68, 'euclidean': 1.13, 'euclidean_l2': 1.13},
                'Dlib': {'cosine': 0.07, 'euclidean': 0.6, 'euclidean_l2': 0.6},
            }
            # Usa um threshold padr�o se o modelo/m�trica n�o estiver mapeado
            threshold = thresholds.get(DEEPFACE_MODEL_NAME, {}).get(DEEPFACE_DISTANCE_METRIC, 0.5)

            # Compara a dist�ncia com o limiar
            if distance <= threshold:
                result_message = f"A pessoa na sua frente parece ser {person_name}."
            else:
                logger.info(f"[DeepFace] Dist�ncia {distance:.4f} > limiar ({threshold}). N�o reconhecido com confian�a.")
                # Poderia retornar o nome com baixa confian�a, ou uma mensagem gen�rica
                result_message = "N�o tenho certeza de quem �, mas detectei um rosto."
                # result_message = f"Detectei um rosto, mas n�o tenho certeza. Pode ser {person_name}?"

        except ValueError as ve: # Captura erro se enforce_detection=True e nenhum rosto for encontrado
            logger.info(f"[DeepFace] Erro (ValueError) ao identificar: {ve}")
            result_message = "N�o detectei um rosto claro para identificar."
        # except ImportError: # J� tratado no in�cio da fun��o
        #      pass
        except Exception as e:
            logger.info(f"[DeepFace] Erro inesperado ao identificar: {e}")
            traceback.print_exc()
            result_message = "Ocorreu um erro ao tentar identificar a pessoa."

        # --- LOGGING FIM ---
        logger.info("[LOG] Finalizado: _handle_identify_person_in_front")
        logger.info(f"[LOG]   - Dura��o: {time.time() - start_time:.2f}s")
        logger.info(f"[LOG]   - Resultado: '{result_message}'")
        # --- FIM LOGGING ---
        return result_message


    def _run_midas_inference(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        # (Fun��o _run_midas_inference inalterada - omitida para brevidade)
        if not self.midas_model or not self.midas_transform:
            logger.info("[MiDaS] Modelo ou transformador n�o carregado.")
            return None
        try:
            # Converte BGR para RGB
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Aplica transforma��es espec�ficas do MiDaS e envia para o dispositivo (CPU/GPU)
            input_batch = self.midas_transform(img_rgb).to(self.midas_device)

            with torch.no_grad(): # Desabilita c�lculo de gradientes para infer�ncia
                prediction = self.midas_model(input_batch)
                # Redimensiona a predi��o para o tamanho original da imagem
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic", # ou "bilinear"
                    align_corners=False,
                ).squeeze()

            # Move o resultado de volta para a CPU e converte para NumPy array
            depth_map = prediction.cpu().numpy()
            return depth_map
        except Exception as e:
            logger.info(f"[MiDaS] Erro durante a infer�ncia: {e}")
            # traceback.print_exc() # Descomente para depura��o detalhada do MiDaS
            return None


    def _find_best_yolo_match(self, object_type: str, yolo_results: List[Any]) -> Optional[Tuple[Dict[str, int], float, str]]:
        # (Fun��o _find_best_yolo_match inalterada - omitida para brevidade)
        best_match = None
        highest_conf = -1.0
        # Obt�m a lista de nomes de classe YOLO correspondentes ao tipo de objeto pedido
        target_yolo_classes = YOLO_CLASS_MAP.get(object_type.lower(), [object_type.lower()])
        # logger.info(f"[YOLO Match] Procurando por classes: {target_yolo_classes}") # Log opcional

        # Verifica se h� resultados YOLO e se o modelo est� carregado
        if not yolo_results or not self.yolo_model:
             # logger.info("[YOLO Match] Sem resultados YOLO ou modelo n�o carregado.") # Log opcional
             return None

        # Itera sobre os resultados (pode haver mais de um se o modelo processar em batch, embora aqui seja 1)
        for result in yolo_results:
            # Verifica se o objeto de resultado tem o atributo 'boxes' e se n�o est� vazio
            if hasattr(result, 'boxes') and result.boxes:
                # Itera sobre cada caixa delimitadora detectada
                for box in result.boxes:
                    # Verifica se a caixa tem os atributos necess�rios
                    if not (hasattr(box, 'cls') and hasattr(box, 'conf') and hasattr(box, 'xyxy')):
                        # logger.info("[YOLO Match] Caixa malformada encontrada.") # Log opcional
                        continue # Pula caixas malformadas

                    # Obt�m ID da classe, confian�a e coordenadas
                    cls_id_tensor = box.cls
                    if cls_id_tensor.nelement() == 0: continue # Tensor vazio
                    cls_id = int(cls_id_tensor[0])

                    conf_tensor = box.conf
                    if conf_tensor.nelement() == 0: continue
                    conf = float(conf_tensor[0])

                    # Obt�m o nome da classe a partir do ID
                    if cls_id < len(self.yolo_model.names):
                        class_name = self.yolo_model.names[cls_id]
                    else:
                        # logger.info(f"[YOLO Match] ID de classe inv�lido: {cls_id}") # Log opcional
                        continue # ID inv�lido

                    # Verifica se a classe detectada � uma das classes alvo
                    if class_name in target_yolo_classes:
                        # Se a confian�a for maior que a melhor encontrada at� agora
                        if conf > highest_conf:
                            highest_conf = conf
                            coords_tensor = box.xyxy[0]
                            if coords_tensor.nelement() < 4: continue # Coordenadas inv�lidas
                            coords = list(map(int, coords_tensor))
                            bbox_dict = {'x1': coords[0], 'y1': coords[1], 'x2': coords[2], 'y2': coords[3]}
                            best_match = (bbox_dict, conf, class_name)
                            logger.info(f"[YOLO Match] Novo melhor match: {class_name} ({conf:.2f})") # Log opcional

        # Retorna a melhor correspond�ncia encontrada (ou None)
        return best_match


    def _estimate_direction(self, bbox: Dict[str, int], frame_width: int) -> str:
        # (Fun��o _estimate_direction inalterada - omitida para brevidade)
        # Calcula o centro horizontal da caixa
        box_center_x = (bbox['x1'] + bbox['x2']) / 2
        # Define a largura da zona central (um ter�o da largura do frame)
        center_zone_width = frame_width / 3

        # Verifica em qual ter�o (esquerda, centro, direita) o centro da caixa est�
        if box_center_x < center_zone_width:
            return "� sua esquerda"
        elif box_center_x > (frame_width - center_zone_width):
            return "� sua direita"
        else:
            return "� sua frente"


    def _check_if_on_surface(self, target_bbox: Dict[str, int], yolo_results: List[Any]) -> bool:
        # (Fun��o _check_if_on_surface inalterada - omitida para brevidade)
        # Define as classes que representam superf�cies de apoio
        surface_classes_keys = ["mesa", "mesa de jantar", "bancada", "prateleira"] # Expandido
        surface_yolo_names = []
        for key in surface_classes_keys:
            surface_yolo_names.extend(YOLO_CLASS_MAP.get(key, [])) # Pega nomes YOLO do mapa
        surface_yolo_names = list(set(surface_yolo_names)) # Remove duplicatas

        if not surface_yolo_names: return False # Se n�o h� classes de superf�cie mapeadas

        # Coordenadas do objeto alvo
        target_bottom_y = target_bbox['y2']
        target_center_x = (target_bbox['x1'] + target_bbox['x2']) / 2

        if not yolo_results or not self.yolo_model:
            return False

        # Itera sobre as detec��es YOLO procurando por superf�cies
        for result in yolo_results:
             if hasattr(result, 'boxes') and result.boxes:
                for box in result.boxes:
                    if not (hasattr(box, 'cls') and hasattr(box, 'xyxy')):
                        continue

                    cls_id_tensor = box.cls
                    if cls_id_tensor.nelement() == 0: continue
                    cls_id = int(cls_id_tensor[0])

                    if cls_id < len(self.yolo_model.names):
                        class_name = self.yolo_model.names[cls_id]
                    else:
                        continue

                    # Se a classe detectada for uma superf�cie
                    if class_name in surface_yolo_names:
                        coords_tensor = box.xyxy[0]
                        if coords_tensor.nelement() < 4: continue
                        s_x1, s_y1, s_x2, s_y2 = map(int, coords_tensor)

                        # --- Heur�stica para verificar se o objeto est� SOBRE a superf�cie ---
                        # 1. Alinhamento Horizontal: O centro X do objeto est� dentro da largura da superf�cie.
                        horizontally_aligned = s_x1 < target_center_x < s_x2

                        # 2. Alinhamento Vertical: A base do objeto (target_bottom_y) est�
                        #    pr�xima ou ligeiramente acima do topo da superf�cie (s_y1).
                        #    Permite uma pequena sobreposi��o ou espa�o.
                        y_tolerance_pixels = 30 # Toler�ncia em pixels (ajustar)
                        # Objeto est� "descansando" perto do topo da superf�cie
                        vertically_aligned = (s_y1 - y_tolerance_pixels) < target_bottom_y < (s_y1 + y_tolerance_pixels * 1.5)

                        # 3. (Opcional) Tamanho Relativo: Evitar que um objeto muito grande "sobre" um pequeno.
                        # target_height = target_bbox['y2'] - target_bbox['y1']
                        # surface_height = s_y2 - s_y1
                        # reasonable_size = target_height < surface_height * 2 # Exemplo

                        # 4. (Opcional) Proximidade: A superf�cie deve estar relativamente pr�xima do objeto.
                        # (Poderia usar MiDaS aqui, mas complica)

                        if horizontally_aligned and vertically_aligned: # and reasonable_size:
                            # logger.info(f"[Surface Check] Objeto em ({target_center_x},{target_bottom_y}) considerado sobre '{class_name}' em ({s_x1}-{s_x2}, {s_y1}-{s_y2})") # Log opcional
                            return True # Encontrou uma superf�cie sob o objeto
        return False # Nenhuma superf�cie encontrada sob o objeto


    def _handle_find_object_and_estimate_distance(self, object_description: str, object_type: str) -> str:
        """Processa a chamada de fun��o para localizar um objeto e estimar dist�ncia."""
        # --- LOGGING IN�CIO ---
        logger.info("[LOG] Executando: _handle_find_object_and_estimate_distance")
        logger.info(f"[LOG]   - Argumentos: object_description='{object_description}', object_type='{object_type}'")
        # --- FIM LOGGING ---

        start_time = time.time()
        result_message = ""

        logger.info(f"[Localizar Objeto] Buscando por '{object_description}' (tipo: '{object_type}')...")
        frame_to_process = None
        yolo_results_for_frame = None
        frame_height, frame_width = 0, 0

        # Obt�m o �ltimo frame e resultados YOLO de forma segura
        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy()
                yolo_results_for_frame = self.latest_yolo_results # Pega os resultados correspondentes
                if frame_to_process is not None:
                    frame_height, frame_width, _ = frame_to_process.shape
            # else: # Frame � None, tratado abaixo

        # Verifica se temos um frame v�lido
        if frame_to_process is None or frame_width == 0 or frame_height == 0:
             logger.info("[Localizar Objeto] Erro: Nenhum frame v�lido dispon�vel.")
             result_message = f"{self.trckuser}, n�o estou enxergando nada no momento para localizar o {object_type}."
             # --- LOGGING FIM ---
             logger.info("[LOG] Finalizado: _handle_find_object_and_estimate_distance (Erro: Sem frame)")
             logger.info(f"[LOG]   - Dura��o: {time.time() - start_time:.2f}s")
             logger.info(f"[LOG]   - Resultado: '{result_message}'")
             # --- FIM LOGGING ---
             return result_message

        # Verifica se temos resultados YOLO para este frame
        if not yolo_results_for_frame:
            logger.info("[Localizar Objeto] Erro: Nenhum resultado YOLO dispon�vel para o frame atual.")
            # Isso pode acontecer se o YOLO falhou ou ainda n�o processou o frame
            result_message = f"{self.trckuser}, n�o consegui processar a imagem a tempo para encontrar o {object_type}."
            # --- LOGGING FIM ---
            logger.info("[LOG] Finalizado: _handle_find_object_and_estimate_distance (Erro: Sem resultados YOLO)")
            logger.info(f"[LOG]   - Dura��o: {time.time() - start_time:.2f}s")
            logger.info(f"[LOG]   - Resultado: '{result_message}'")
            # --- FIM LOGGING ---
            return result_message

        # Encontra a melhor correspond�ncia YOLO para o tipo de objeto
        best_yolo_match = self._find_best_yolo_match(object_type, yolo_results_for_frame)

        # Fallback: Se n�o encontrou pelo tipo, tenta pela �ltima palavra da descri��o
        if not best_yolo_match:
            logger.info(f"[Localizar Objeto] Nenhum objeto do tipo '{object_type}' encontrado. Tentando fallback com descri��o...")
            last_word = object_description.split(" ")[-1].lower()
            # Evita tentar o mesmo tipo duas vezes se type j� era a �ltima palavra
            if last_word != object_type.lower():
                target_yolo_classes_fallback = YOLO_CLASS_MAP.get(last_word, [last_word])
                logger.info(f"[Localizar Objeto] Fallback: Buscando por classes: {target_yolo_classes_fallback}")
                best_yolo_match = self._find_best_yolo_match(last_word, yolo_results_for_frame)

            # Se ainda n�o encontrou ap�s o fallback
            if not best_yolo_match:
                 logger.info(f"[Localizar Objeto] Objeto '{object_description}' n�o encontrado mesmo com fallback.")
                 result_message = f"{self.trckuser}, n�o consegui encontrar um(a) {object_description} na imagem."
                 # --- LOGGING FIM ---
                 logger.info("[LOG] Finalizado: _handle_find_object_and_estimate_distance (N�o encontrado)")
                 logger.info(f"[LOG]   - Dura��o: {time.time() - start_time:.2f}s")
                 logger.info(f"[LOG]   - Resultado: '{result_message}'")
                 # --- FIM LOGGING ---
                 return result_message

        # Se encontrou um objeto
        target_bbox, confidence, detected_class = best_yolo_match
        logger.info(f"[Localizar Objeto] Melhor correspond�ncia YOLO: Classe '{detected_class}', Conf: {confidence:.2f}, BBox: {target_bbox}")

        # Verifica se est� sobre uma superf�cie
        is_on_surface = self._check_if_on_surface(target_bbox, yolo_results_for_frame)
        surface_msg = "sobre uma superf�cie (como uma mesa)" if is_on_surface else ""

        # Estima a dire��o
        direction = self._estimate_direction(target_bbox, frame_width)

        # Estima a dist�ncia usando MiDaS
        distance_steps = -1
        depth_map = None
        if self.midas_model:
            logger.info("[Localizar Objeto] Executando MiDaS...")
            depth_map = self._run_midas_inference(frame_to_process) # Executa infer�ncia MiDaS
        else:
            logger.info("[Localizar Objeto] MiDaS n�o dispon�vel. N�o � poss�vel estimar dist�ncia.")

        if depth_map is not None:
            try:
                # Pega o valor de profundidade no centro da caixa do objeto
                center_x = int((target_bbox['x1'] + target_bbox['x2']) / 2)
                center_y = int((target_bbox['y1'] + target_bbox['y2']) / 2)
                # Garante que as coordenadas est�o dentro dos limites do mapa de profundidade
                center_y = max(0, min(center_y, depth_map.shape[0] - 1))
                center_x = max(0, min(center_x, depth_map.shape[1] - 1))
                depth_value = depth_map[center_y, center_x]

                # --- Heur�stica de Convers�o MiDaS (MUITO BRUTA - PRECISA CALIBRAR) ---
                # MiDaS_small retorna profundidade inversa (maior valor = mais perto)
                # Valores dependem muito da escala da cena e do modelo.
                # Esta � uma tentativa de mapeamento muito simplificada.
                if depth_value > 1e-6: # Evita divis�o por zero ou valores inv�lidos
                    # Mapeamento heur�stico (ajustar com base em testes reais)
                    # Exemplo: depth > 250 -> muito perto; depth < 30 -> longe
                    if depth_value > 300:  # Ajuste estes valores!
                        estimated_meters = np.random.uniform(0.5, 1.5) # Ex: 0.5-1.5m
                    elif depth_value > 150: # Ajuste!
                        estimated_meters = np.random.uniform(1.5, 3.5) # Ex: 1.5-3.5m
                    elif depth_value > 50:  # Ajuste!
                        estimated_meters = np.random.uniform(3.5, 7.0) # Ex: 3.5-7m
                    else: # Longe
                        estimated_meters = np.random.uniform(7.0, 15.0) # Ex: 7-15m

                    # Limita a estimativa a um alcance razo�vel
                    estimated_meters = max(0.5, min(estimated_meters, 20))
                    # Converte metros para passos
                    distance_steps = max(1, round(estimated_meters / METERS_PER_STEP)) # Garante pelo menos 1 passo
                    logger.info(f"[Localizar Objeto] Profundidade MiDaS no centro ({center_y},{center_x}): {depth_value:.4f}, Metros Estimados (heur�stico): {estimated_meters:.2f}, Passos: {distance_steps}")
                else:
                     logger.info("[Localizar Objeto] Valor de profundidade MiDaS inv�lido ou muito baixo no centro do objeto.")
            except IndexError:
                logger.info(f"[Localizar Objeto] Erro: Coordenadas ({center_y},{center_x}) fora dos limites do mapa de profundidade ({depth_map.shape}).")
            except Exception as e_depth:
                logger.info(f"[Localizar Objeto] Erro ao extrair/processar profundidade do MiDaS: {e_depth}")
                # traceback.print_exc() # Descomente para depura��o
                distance_steps = -1 # Reseta se houve erro
        # else: MiDaS n�o dispon�vel ou falhou, distance_steps continua -1

        # --- Constr�i a Mensagem de Resposta ---
        # Usa a descri��o original do usu�rio para a resposta
        object_name_for_response = object_description

        response_parts = [f"{self.trckuser}, o {object_name_for_response} est�"]
        if surface_msg:
            response_parts.append(surface_msg) # Adiciona "sobre uma superf�cie..."

        if distance_steps > 0:
            # Adiciona a dist�ncia em passos
            response_parts.append(f"a aproximadamente {distance_steps} passo{'s' if distance_steps > 1 else ''}")

        # Adiciona a dire��o (sempre)
        response_parts.append(direction)

        # Junta as partes da resposta
        if len(response_parts) > 1: # Se adicionou algo al�m de "Usu�rio, o obj est�"
            # Junta com v�rgulas, exceto antes da dire��o final
            if len(response_parts) > 2:
                # Ex: "..., sobre a superf�cie, a X passos, � sua frente."
                result_message = ", ".join(response_parts[:-1]) + " " + response_parts[-1] + "."
            else:
                # Ex: "..., a X passos � sua frente." ou "..., sobre a superf�cie � sua frente."
                result_message = " ".join(response_parts) + "."
        else:
            # Caso muito raro onde s� temos a dire��o (sem dist�ncia e sem superf�cie)
            result_message = f"{self.trckuser}, o {object_name_for_response} est� {direction}."


        # --- LOGGING FIM ---
        logger.info("[LOG] Finalizado: _handle_find_object_and_estimate_distance")
        logger.info(f"[LOG]   - Dura��o: {time.time() - start_time:.2f}s")
        logger.info(f"[LOG]   - Resultado: '{result_message}'")
        # --- FIM LOGGING ---
        return result_message


    async def receive_audio(self):
        # (Fun��o receive_audio inalterada, exceto por traceback e coment�rio - omitida para brevidade)
        logger.info("Receive_audio pronto para receber respostas do Gemini...")
        try:
            if not self.session:
                logger.info("Sess�o Gemini n�o estabelecida em receive_audio. Encerrando.")
                self.stop_event.set()
                return

            while not self.stop_event.is_set():
                # Verifica se a sess�o ainda existe (pode ser fechada por outra tarefa)
                if not self.session:
                    logger.info("Sess�o Gemini desconectada em receive_audio. Aguardando reconex�o ou parada.")
                    await asyncio.sleep(1)
                    if not self.session and not self.stop_event.is_set(): # Verifica de novo
                        logger.info("Sess�o ainda n�o dispon�vel. Sinalizando parada para reconex�o.")
                        self.stop_event.set() # Sinaliza para o loop run tentar reconectar
                    elif self.session:
                        logger.info("Sess�o Gemini reconectada.")
                    break # Sai do loop interno para o run tentar reconectar ou parar


                try:
                    has_received_data_in_turn = False
                    # logger.info("Aguardando pr�ximo turno de resposta do Gemini...") # Log opcional

                    # NOTA: Este loop `async for` n�o tem um timeout expl�cito.
                    # Se a API travar ou parar de enviar dados sem fechar o stream,
                    # esta tarefa pode bloquear indefinidamente. A detec��o de erros
                    # abaixo (LiveSession closed, etc.) mitiga isso parcialmente.
                    async for response_part in self.session.receive():
                   #     logger.info(f"--- RAW GEMINI RESPONSE PART ---")
                    #    logger.info(f"Response_part.text: {response_part.text if hasattr(response_part, 'text') else 'N/A'}")
                     #   logger.info(f"Response_part.function_call: {response_part.function_call if hasattr(response_part, 'function_call') else 'N/A'}")
                      #  logger.info(f"Response_part.data (presente?): {hasattr(response_part, 'data') and bool(response_part.data)}")
                       # logger.info(f"Awaiting name flag: {self.awaiting_name_for_save_face}")
                        #logger.info(f"Thinking event set: {self.thinking_event.is_set()}")
                        if self.stop_event.is_set():
                            break
                        if has_received_data_in_turn and response_part.text:
                            logger.info("Descartando texto antigo para nova entrada.")
                            continue
                        has_received_data_in_turn = True

                        if self.stop_event.is_set():
                            logger.info("Sinal de parada recebido durante processamento de resposta.")
                            break

                        # --- Processa �udio ---
                        if response_part.data:
                            if self.audio_in_queue:
                                try:
                                    self.audio_in_queue.put_nowait(response_part.data)
                                except asyncio.QueueFull:
                                    logger.info("Aviso: Fila de �udio de entrada cheia. �udio descartado.")
                            continue # Processou �udio, vai para pr�xima parte

                        # --- Processa Nome Pendente (Fluxo save_known_face) ---
                        if self.awaiting_name_for_save_face:
                            user_provided_name = None
                            if response_part.text: # Gemini transcreveu fala ou usu�rio digitou
                                user_provided_name = response_part.text.strip()
                                logger.info(f"[Trackie] Recebido texto enquanto aguardava nome: '{user_provided_name}'")

                            if user_provided_name:
                                logger.info(f"[Trackie] Processando nome '{user_provided_name}' para salvar rosto...")
                                self.awaiting_name_for_save_face = False # Reseta a flag

                                original_function_name_pending = "save_known_face" # Nome da fun��o original

                                logger.info("Pensando...") # Feedback visual
                                self.thinking_event.set() # Pausa envio de dados

                                # Feedback de voz ANTES de executar a fun��o
                                voice_feedback_msg = f"{self.trckuser}, salvando rosto de {user_provided_name}, um momento..."
                                if self.session:
                                    try:
                                        # Envia feedback e termina o turno da IA para que ela fale
                                        await self.session.send(input=voice_feedback_msg, end_of_turn=True)
                                        logger.info(f"  [Feedback Enviado]: {voice_feedback_msg}")
                                    except Exception as e_feedback:
                                        logger.error(f"Erro ao enviar feedback de voz (awaiting name): {e_feedback}")

                                # Executa a fun��o s�ncrona em outra thread
                                result_message = await asyncio.to_thread(self._handle_save_known_face, user_provided_name)

                                # Envia o resultado da fun��o de volta para o Gemini
                                logger.info(f"  [Trackie] Resultado da Fun��o '{original_function_name_pending}': {result_message}")
                                if self.session:
                                    try:
                                        await self.session.send(
                                            input=types.Content(
                                                role="tool", # Importante: role="tool" para respostas de fun��o
                                                parts=[types.Part.from_function_response(
                                                    name=original_function_name_pending,
                                                    response={"result": Value(string_value=result_message)}
                                                )]
                                            )
                                            # N�o usar end_of_turn=True aqui, deixa Gemini decidir quando responder
                                        )
                                        logger.info("  [Trackie] Resultado da fun��o (awaiting name) enviado.")
                                    except Exception as e_send_fc_resp:
                                        logger.error(f"Erro ao enviar FunctionResponse (awaiting name): {e_send_fc_resp}")
                                else:
                                    logger.info("  [Trackie] Sess�o inativa. N�o foi poss�vel enviar resultado da fun��o (awaiting name).")

                                # Libera o envio de dados
                                if self.thinking_event.is_set():
                                    self.thinking_event.clear()
                                logger.info("Pensamento conclu�do (awaiting name).")
                                continue # Processamos este input, vamos para o pr�ximo response_part

                        # --- Processa Texto da IA ---
                        if response_part.text:
                            # Imprime texto recebido do Gemini (pode ser parcial)
                            # Usando print() diretamente para garantir que o 'end=""' funcione como esperado no console.
                            print(f"\n[Gemini Texto]: {response_part.text}", end="")
                            # logger.info(f"\n[Gemini Texto]: {response_part.text}", extra={'end': ''}) # Alternativa se o logger for configurado para lidar com 'end'


                        # --- Processa Chamada de Fun��o ---
                        if getattr(response_part, "function_call", None):
                            fc = response_part.function_call
                            function_name = fc.name
                            args = {key: val for key, val in fc.args.items()}
                            logger.info(f"\n[Gemini Function Call] Recebido: {function_name}, Args: {args}")

                            result_message = None # Inicializa resultado

                            # --- Caso Especial: save_known_face sem nome ---
                            if function_name == "save_known_face" and not args.get("person_name"):
                                self.awaiting_name_for_save_face = True # Ativa a flag
                                if self.thinking_event.is_set(): # Garante que n�o est� pensando enquanto pergunta
                                    self.thinking_event.clear()
                                logger.info("[Trackie] Nome n�o fornecido para save_known_face. Solicitando ao usu�rio.")
                                # Pede o nome ao usu�rio via Gemini (voz)
                                if self.session:
                                    try:
                                        # Envia a pergunta e termina o turno da IA
                                        await self.session.send(input=f"{self.trckuser}, por favor forne�a o nome da pessoa para salvar o rosto.", end_of_turn=True)
                                    except Exception as e_ask_name:
                                        logger.error(f"Erro ao pedir nome para save_face: {e_ask_name}")
                                # N�o executa a fun��o local nem envia FC response agora

                            # --- Caso Normal: Outras fun��es ou save_known_face com nome ---
                            else:
                                logger.info("Pensando...") # Feedback visual
                                self.thinking_event.set() # Pausa envio

                                # Monta mensagem de feedback de voz
                                voice_feedback_msg = f"{self.trckuser}, processando {function_name}, um momento..." # Padr�o
                                if function_name == "save_known_face":
                                    person_name_fb = args.get('person_name', 'pessoa')
                                    voice_feedback_msg = f"{self.trckuser}, salvando rosto de {person_name_fb}, um momento..."
                                    logger.info(f"Salvando rosto de {person_name_fb} ")
                                elif function_name == "identify_person_in_front":
                                    voice_feedback_msg = "Usu�rio, identificando pessoa, um momento..."
                                    logger.info(f"Identificando rosto... {person_name_fb} ")
                                elif function_name == "find_object_and_estimate_distance":
                                    obj_desc_fb = args.get('object_description', 'objeto')
                                    voice_feedback_msg = f"{self.trckuser}, localizando {obj_desc_fb}, um momento..."
                                    logger.info(f"Salvando rosto de {person_name_fb} ")

                                # Envia feedback de voz ANTES de executar a fun��o
                                if self.session:
                                    try:
                                        # Envia feedback e termina o turno da IA
                                        await self.session.send(input=voice_feedback_msg, end_of_turn=True)
                                        logger.info(f"  [Feedback Enviado]: {voice_feedback_msg}")
                                    except Exception as e_feedback:
                                        logger.error(f"Erro ao enviar feedback pr�-fun��o: {e_feedback}")

                                # --- Executa a Fun��o Local Correspondente ---
                                # Verifica se a fun��o requer modo c�mera
                                vision_functions = ["save_known_face", "identify_person_in_front", "find_object_and_estimate_distance"]
                                if self.video_mode != "camera" and function_name in vision_functions:
                                    logger.info(f"[Function Call] '{function_name}' requer modo c�mera, mas modo atual � '{self.video_mode}'.")
                                    result_message = "Desculpe, esta fun��o s� est� dispon�vel quando a c�mera est� ativa."
                                else:
                                    logger.info(f"  [Trackie] Processando Fun��o '{function_name}' em background...")
                                    # Chama o handler apropriado em outra thread
                                    try:
                                        if function_name == "save_known_face":
                                            person_name_arg = args.get("person_name")
                                            if person_name_arg:
                                                result_message = await asyncio.to_thread(self._handle_save_known_face, person_name_arg)
                                            else:
                                                result_message = "Erro interno: nome n�o dispon�vel para salvar rosto neste ponto."
                                                logger.error("ERRO L�GICO: Tentativa de chamar _handle_save_known_face sem nome.")
                                        elif function_name == "identify_person_in_front":
                                            # Verifica se pandas est� dispon�vel
                                            if pd is None:
                                                 result_message = "Erro interno: depend�ncia 'pandas' faltando para identifica��o."
                                            else:
                                                result_message = await asyncio.to_thread(self._handle_identify_person_in_front)

                                        elif function_name == "find_object_and_estimate_distance":
                                            desc = args.get("object_description")
                                            obj_type = args.get("object_type")
                                            if desc and obj_type:
                                                # Verifica se MiDaS est� funcional antes de chamar
                                                if not self.midas_model:
                                                    result_message = "Usu�rio, desculpe, o m�dulo de estimativa de dist�ncia n�o est� funcionando no momento."
                                                else:
                                                    result_message = await asyncio.to_thread(
                                                        self._handle_find_object_and_estimate_distance, desc, obj_type
                                                    )
                                            else:
                                                result_message = "Descri��o ou tipo do objeto n�o fornecido corretamente para localiza��o."
                                                logger.error(f"ERRO: Argumentos faltando para find_object_and_estimate_distance: desc='{desc}', type='{obj_type}'")
                                        else:
                                            result_message = f"Fun��o '{function_name}' desconhecida ou n�o implementada."
                                            logger.warning(f"AVISO: Recebida chamada para fun��o n�o mapeada: {function_name}")
                                    except Exception as e_handler:
                                         logger.error(f"Erro ao executar handler para '{function_name}': {e_handler}")
                                         traceback.print_exc()
                                         result_message = f"Ocorreu um erro interno ao processar a fun��o {function_name}."


                            # --- Envia Resultado da Fun��o de Volta (se houver) ---
                            if result_message is not None: # S� envia se um resultado foi gerado
                                logger.info(f"  [Trackie] Resultado da Fun��o '{function_name}': {result_message}")
                                if self.session:
                                    try:
                                        await self.session.send(
                                            input=types.Content(
                                                role="tool", # Importante
                                                parts=[types.Part.from_function_response(
                                                    name=function_name,
                                                    response={"result": Value(string_value=result_message)}
                                                )]
                                            )
                                        )
                                        logger.info("  [Trackie] Resultado da fun��o enviado.")
                                    except Exception as e_send_fc_resp_main:
                                        logger.error(f"Erro ao enviar FunctionResponse (main): {e_send_fc_resp_main}")
                                else:
                                    logger.info("  [Trackie] Sess�o inativa. N�o foi poss�vel enviar resultado da fun��o.")

                                # Libera o envio de dados ap�s processar a fun��o
                                if self.thinking_event.is_set():
                                     self.thinking_event.clear()
                                logger.info("Pensamento conclu�do (function call).")
                            # Se result_message � None (caso de pedir nome), thinking_event j� foi limpo antes

                    # --- Fim do processamento de um turno da IA ---
                    if not self.stop_event.is_set():
                        if has_received_data_in_turn:
                            # logger.info("\nFim do turno de resposta do Gemini.") # Log opcional
                            # Adiciona uma nova linha se o �ltimo log foi um texto parcial do Gemini
                            if response_part and response_part.text and not response_part.text.endswith('\n'):
                                print() # Garante que o pr�ximo log comece em uma nova linha
                            pass # Continua esperando o pr�ximo turno
                        else:
                            # Se o loop `async for` terminar sem dados, pode indicar fim normal ou problema
                            # logger.info("Stream do turno atual terminou sem dados.") # Log opcional
                            await asyncio.sleep(0.05) # Pequena pausa
                    if self.stop_event.is_set():
                        break # Sai do loop `async for` se stop foi chamado

                # --- Tratamento de Erros no Loop de Recebimento ---
                except Exception as e_inner_loop:
                    logger.error(f"Erro durante o recebimento/processamento de resposta: {e_inner_loop}")
                    error_str_upper = str(e_inner_loop).upper()
                    # Verifica erros que indicam sess�o fechada/perdida
                    if "LIVESESSION CLOSED" in error_str_upper or \
                       "LIVESESSION NOT CONNECTED" in error_str_upper or \
                       "DEADLINE EXCEEDED" in error_str_upper or \
                       "RST_STREAM" in error_str_upper or \
                       "UNAVAILABLE" in error_str_upper:
                        logger.info("Erro indica que a sess�o Gemini foi fechada/perdida. Sinalizando parada.")
                        self.stop_event.set()
                        break # Sai do loop while principal
                    else:
                        # Outros erros podem ser tempor�rios ou bugs
                        traceback.print_exc()
                        await asyncio.sleep(0.5) # Pausa antes de tentar continuar

            # Se o loop while terminar por causa do stop_event
            if self.stop_event.is_set():
                logger.info("Loop de recebimento de �udio interrompido pelo stop_event.")

        except asyncio.CancelledError:
            logger.info("receive_audio foi cancelado.")
        except Exception as e:
            # Erro cr�tico fora do loop principal (ex: na configura��o inicial)
            logger.error(f"Erro cr�tico em receive_audio: {e}")
            traceback.print_exc()
            self.stop_event.set() # Garante que tudo pare
        finally:
            logger.info("receive_audio finalizado.")
            # Limpa a flag caso a tarefa seja cancelada enquanto esperava nome
            self.awaiting_name_for_save_face = False
            # Garante que thinking_event seja limpo na sa�da
            if self.thinking_event.is_set():
                self.thinking_event.clear()


    async def play_audio(self):
        # (Fun��o play_audio inalterada, exceto por traceback em erro cr�tico - omitida para brevidade)
        if not pya:
            logger.info("PyAudio n�o inicializado. Tarefa play_audio n�o pode iniciar.")
            return

        stream = None
        output_rate = RECEIVE_SAMPLE_RATE # Taxa padr�o
        try:
            logger.info("Configurando stream de �udio de sa�da...")
            try:
                # Tenta usar a taxa de amostragem do dispositivo padr�o
                out_device_info = pya.get_default_output_device_info()
                logger.info(f"Usando dispositivo de sa�da: {out_device_info['name']} @ {output_rate} Hz")
            except Exception as e_dev_info:
                logger.info(f"N�o foi poss�vel obter info do dispositivo de sa�da padr�o ({e_dev_info}). Usando taxa padr�o: {output_rate} Hz")

            # Abre o stream de sa�da em outra thread
            stream = await asyncio.to_thread(
                pya.open, format=FORMAT, channels=CHANNELS, rate=output_rate, output=True
            )
            logger.info("Player de �udio pronto.")

            while not self.stop_event.is_set():
                # Verifica se a fila de entrada existe
                if not self.audio_in_queue:
                    await asyncio.sleep(0.1)
                    continue

                # Tenta obter �udio da fila com timeout
                try:
                    if self.out_queue and not self.out_queue.empty():
                        logger.info("Nova entrada detectada, interrompendo �udio atual.")
                        stream.stop_stream()
                        await asyncio.sleep(0.1)
                        stream.start_stream()
                    bytestream = await asyncio.wait_for(self.audio_in_queue.get(), timeout=0.5)

                    if bytestream is None: # Sinal de parada da fila
                        logger.info("Recebido sinal de encerramento (None) para play_audio.")
                        break # Sai do loop while

                    # Verifica se o stream est� ativo e escreve os dados
                    if stream and stream.is_active():
                        # Escreve no stream em outra thread
                        await asyncio.to_thread(stream.write, bytestream)
                    else:
                        logger.info("Stream de �udio para playback n�o est� ativo. Descartando �udio.")
                        # N�o marca task_done se descartou? Ou marca?
                        # Melhor marcar para n�o bloquear a fila caso o stream morra.
                        if self.audio_in_queue: self.audio_in_queue.task_done()


                    # Marca a tarefa como conclu�da na fila
                    if self.audio_in_queue: self.audio_in_queue.task_done()

                except asyncio.TimeoutError:
                    # Timeout � normal se n�o houver �udio por 0.5s
                    continue
                except asyncio.QueueEmpty:
                    # Fila vazia tamb�m � normal
                    continue
                except OSError as e_os_play:
                    # Erro comum se o stream for fechado inesperadamente
                    if "Stream closed" in str(e_os_play):
                        logger.info("Stream de playback fechado (OSError). Encerrando play_audio.")
                        break # Sai do loop while
                    else:
                        logger.error(f"Erro de OS ao reproduzir �udio: {e_os_play}")
                        traceback.print_exc()
                        break # Sai em outros erros de OS tamb�m
                except Exception as e_inner:
                    # Erro gen�rico durante a reprodu��o
                    logger.error(f"Erro ao reproduzir �udio (interno): {e_inner}")
                    if "Stream closed" in str(e_inner): # Verifica se o erro indica stream fechado
                        logger.info("Stream de playback fechado (Exception). Encerrando play_audio.")
                        break # Sai do loop while
                    traceback.print_exc()
                    # Decide se continua ou para em outros erros
                    # break

        except asyncio.CancelledError:
            logger.info("play_audio foi cancelado.")
        except Exception as e:
            # Erro cr�tico na configura��o ou loop principal
            logger.error(f"Erro cr�tico em play_audio: {e}")
            traceback.print_exc()
            # N�o seta stop_event aqui, deixa o run gerenciar
        finally:
            logger.info("Finalizando play_audio...")
            # Garante que o stream seja fechado
            if stream:
                try:
                    # Espera o buffer esvaziar antes de fechar (opcional)
                    # await asyncio.to_thread(stream.stop_stream)
                    if stream.is_active():
                         stream.stop_stream()
                    stream.close()
                    logger.info("Stream de �udio de sa�da fechado.")
                except Exception as e_close:
                    logger.error(f"Erro ao fechar stream de �udio de sa�da: {e_close}")
            logger.info("play_audio conclu�do.")


    async def run(self):
        # (Fun��o run inalterada, exceto por traceback em erro cr�tico - omitida para brevidade)
        logger.info("Iniciando AudioLoop...")
        max_retries = 3
        retry_delay_base = 2.0 # Aumentado ligeiramente o delay base

        attempt = 0
        while attempt <= max_retries and not self.stop_event.is_set():
            retry_delay = retry_delay_base * (2 ** attempt) # Backoff exponencial
            try:
                # Se for uma tentativa de reconex�o, espera
                if attempt > 0:
                     logger.info(f"Tentativa de reconex�o {attempt}/{max_retries} ap�s {retry_delay:.1f}s...")
                     await asyncio.sleep(retry_delay)

                # --- Limpa estado da sess�o anterior ---
                # Garante que a sess�o antiga seja fechada se existir
                if self.session:
                    try:
                        await self.session.close()
                    except Exception: pass # Ignora erros ao fechar sess�o antiga
                self.session = None
                self.audio_in_queue = None # Ser� recriado
                self.out_queue = None      # Ser� recriado
                self.awaiting_name_for_save_face = False # Reseta estado da fun��o
                if self.thinking_event.is_set(): # Garante que n�o comece pensando
                    self.thinking_event.clear()
                # --- Fim da Limpeza ---


                # --- Tenta conectar ---
                # Verifica se o cliente Gemini foi inicializado com sucesso
                if client is None:
                    logger.info("ERRO: Cliente Gemini n�o inicializado. N�o � poss�vel conectar.")
                    self.stop_event.set()
                    break

                logger.info("Tentando conectar ao Gemini (Tentativa {attempt+1})...")
                async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                    self.session = session
                    session_id_str = 'N/A'
                    # Tenta obter o ID da sess�o se o atributo existir
                    if hasattr(session, 'session_id'):
                         session_id_str = session.session_id
                    elif hasattr(session, '_session_id'): # Tenta atributo privado como fallback
                         session_id_str = session._session_id

                    logger.info(f"Sess�o Gemini LiveConnect estabelecida (Tentativa {attempt+1}). ID: {session_id_str}")
                    attempt = 0 # Reseta tentativas em caso de sucesso

                    # Recria filas para a nova sess�o
                    self.audio_in_queue = asyncio.Queue()
                    # Aumenta tamanho da fila de sa�da para acomodar bursts
                    self.out_queue = asyncio.Queue(maxsize=150)

                    # --- Inicia todas as tarefas da sess�o ---
                    async with asyncio.TaskGroup() as tg:
                        logger.info("Iniciando tarefas da sess�o...")
                        # Tarefa para ler input de texto do console
                        tg.create_task(self.send_text(), name="send_text_task")
                        # Tarefa para enviar dados (�udio/v�deo) da out_queue para Gemini
                        tg.create_task(self.send_realtime(), name="send_realtime_task")
                        # Tarefa para capturar �udio do microfone (se PyAudio estiver ok)
                        if pya: tg.create_task(self.listen_audio(), name="listen_audio_task")

                        # Tarefa para capturar v�deo (c�mera ou tela)
                        if self.video_mode == "camera":
                            tg.create_task(self.get_frames(), name="get_frames_task")
                        elif self.video_mode == "screen":
                            tg.create_task(self.get_screen(), name="get_screen_task")
                        # Se mode for "none", nenhuma tarefa de v�deo � iniciada

                        # Tarefa para receber e processar respostas (�udio/texto/FC) do Gemini
                        tg.create_task(self.receive_audio(), name="receive_audio_task")
                        # Tarefa para tocar �udio recebido do Gemini (se PyAudio estiver ok)
                        if pya: tg.create_task(self.play_audio(), name="play_audio_task")

                        logger.info("Todas as tarefas da sess�o iniciadas. Aguardando conclus�o ou parada...")
                    # O bloco `async with tg:` espera todas as tarefas terminarem

                    logger.info("TaskGroup da sess�o finalizado.")
                    # Se o TaskGroup terminou sem o stop_event ser setado,
                    # significa que a sess�o Gemini provavelmente fechou ou uma tarefa cr�tica falhou.
                    if not self.stop_event.is_set():
                         logger.info("Sess�o Gemini terminou inesperadamente ou TaskGroup conclu�do. Tentando reconectar...")
                         attempt += 1 # Incrementa para tentar reconectar
                    else:
                        # Se stop_event foi setado, sa�mos do loop principal
                        logger.info("Stop event detectado ap�s TaskGroup. Encerrando loop de conex�o.")
                        break


            except asyncio.CancelledError:
                logger.info("Loop principal (run) cancelado.")
                self.stop_event.set() # Garante que o evento de parada seja definido
                break
            except ExceptionGroup as eg:
                # Erro vindo do TaskGroup (uma ou mais tarefas falharam)
                logger.error(f"Erro(s) no TaskGroup (Tentativa {attempt+1}):")
                self.stop_event.set() # Para tudo se uma tarefa falhar criticamente
                for i, exc in enumerate(eg.exceptions):
                    logger.info(f"  Erro {i+1}: {type(exc).__name__} - {exc}")
                    # Imprime traceback para cada exce��o no grupo
                    # traceback.print_exception(type(exc), exc, exc.__traceback__)
                attempt += 1 # Tenta reconectar ap�s falha no TaskGroup
                self.session = None # Garante que a sess�o seja considerada inv�lida
            except Exception as e:
                # Erro durante a conex�o inicial ou outro erro inesperado no loop run
                logger.error(f"Erro ao conectar ou erro inesperado no m�todo run (Tentativa {attempt+1}): {type(e).__name__} - {e}")
                traceback.print_exc() # Imprime traceback completo

                # Verifica se o erro � relacionado � conex�o/sess�o para decidir se retenta
                error_str_upper = str(e).upper()
                # Adiciona mais verifica��es de strings comuns de erro de conex�o/gRPC
                is_connection_error = any(err_str in error_str_upper for err_str in [
                    "RST_STREAM", "UNAVAILABLE", "DEADLINE_EXCEEDED",
                    "LIVESESSION CLOSED", "LIVESESSION NOT CONNECTED",
                    "CONNECTIONCLOSEDERROR", "GOAWAY", "INTERNALERROR",
                    "FAILED TO ESTABLISH CONNECTION", "AUTHENTICATION" # Adiciona erro de autentica��o
                ])

                if is_connection_error:
                    logger.info(f"Detectado erro relacionado � sess�o ou conex�o Gemini: {e}")
                else:
                    logger.info("Erro n�o parece ser diretamente de conex�o. Verifique o traceback.")

                attempt += 1 # Incrementa tentativa
                self.session = None # Garante que a sess�o seja considerada inv�lida
                if attempt > max_retries:
                     logger.info("M�ximo de tentativas de reconex�o atingido ap�s erro. Encerrando.")
                     self.stop_event.set() # Define parada ap�s exceder retries
                     break # Sai do loop while

        # --- Fim do Loop de Conex�o ---
        if not self.stop_event.is_set() and attempt > max_retries:
             logger.info("N�o foi poss�vel restabelecer a conex�o com Gemini ap�s m�ltiplas tentativas.")
             self.stop_event.set() # Garante que o evento de parada esteja definido

        # --- Limpeza Final ---
        logger.info("Iniciando limpeza final em AudioLoop.run()...")
        self.stop_event.set() # Garante que todas as tarefas saibam que devem parar

        # Fecha a sess�o Gemini se ainda estiver ativa
        if self.session:
            try:
                logger.info("Fechando sess�o LiveConnect ativa...")
                await self.session.close()
                logger.info("Sess�o LiveConnect fechada.")
            except Exception as e_close_session:
                logger.error(f"Erro ao fechar sess�o LiveConnect na limpeza final: {e_close_session}")
        self.session = None

        # Sinaliza para a tarefa play_audio parar (colocando None na fila)
        if self.audio_in_queue:
            try:
                # N�o espera se a fila estiver cheia, apenas tenta colocar
                self.audio_in_queue.put_nowait(None)
            except asyncio.QueueFull:
                logger.info("Aviso: N�o foi poss�vel colocar None na audio_in_queue (cheia) durante a limpeza.")
            except Exception as e_q_put:
                logger.error(f"Erro ao colocar None na audio_in_queue durante a limpeza: {e_q_put}")

        # Fecha janelas OpenCV se estiverem ativas
        if self.preview_window_active:
            logger.info("Fechando janelas OpenCV...")
            try:
                cv2.destroyAllWindows()
                logger.info("Janelas OpenCV destru�das no finally de run.")
            except Exception as e_cv_destroy_all:
                 logger.warning(f"AVISO: erro ao tentar fechar janelas de preview na limpeza final: {e_cv_destroy_all}")
            self.preview_window_active = False

        # Termina PyAudio
        if pya:
            try:
                logger.info("Terminando PyAudio...")
                # N�o precisa chamar stop_stream/close aqui, pois play_audio/listen_audio j� o fazem
                pya.terminate()
                logger.info("Recursos de PyAudio liberados.")
            except Exception as e_pya:
                logger.error(f"Erro ao terminar PyAudio: {e_pya}")
        logger.info("Limpeza de AudioLoop.run() conclu�da.")


# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trackie - Assistente IA visual e auditivo.")
    parser.add_argument(
        "--mode", type=str, default=DEFAULT_MODE, choices=["camera", "screen", "none"],
        help="Modo de opera��o para entrada de v�deo/imagem ('camera', 'screen', 'none')."
    )
    parser.add_argument(
        "--show_preview", action="store_true",
        help="Mostra janela com preview da c�mera e detec��es YOLO (apenas no modo 'camera')."
    )
    args = parser.parse_args()

    # Valida o argumento show_preview
    show_actual_preview = False
    if args.mode == "camera" and args.show_preview:
        show_actual_preview = True
        logger.info("Feedback visual da c�mera (preview) ATIVADO.")
    elif args.mode != "camera" and args.show_preview:
        logger.info("Aviso: --show_preview s� tem efeito com --mode camera. Ignorando.")
    else:
        logger.info("Feedback visual da c�mera (preview) DESATIVADO.")


    # Verifica se o modelo YOLO existe se o modo for camera
    if args.mode == "camera":
        if not os.path.exists(YOLO_MODEL_PATH):
            logger.error(f"ERRO CR�TICO: Modelo YOLO '{YOLO_MODEL_PATH}' N�o encontrado.")
            logger.info("Verifique o caminho em BaseDir e YOLO_MODEL_PATH ou baixe o modelo.")
            exit(1) # Sai se o modelo n�o for encontrado

    # Verifica se PyAudio foi inicializado
    if not pya:
         logger.info("ERRO CR�TICO: PyAudio n�o p�de ser inicializado.")
         logger.info("Verifique a instala��o do PyAudio e suas depend�ncias (como PortAudio).")
         logger.info("O programa n�o pode funcionar sem �udio. Encerrando.")
         exit(1) # Sai se PyAudio falhou

    # Verifica se o cliente Gemini foi inicializado
    if client is None:
        logger.info("ERRO CR�TICO: Cliente Gemini n�o p�de ser inicializado (verifique API Key/conex�o). Encerrando.")
        exit(1)

    # Verifica se o arquivo de prompt foi carregado (system_instruction_text deve existir)
    if 'system_instruction_text' not in globals() or not system_instruction_text or system_instruction_text == "Voc� � um assistente prestativo.":
         logger.info("AVISO: Falha ao carregar a instru��o do sistema do arquivo ou arquivo n�o encontrado. Usando prompt padr�o.")
         # Decide se quer continuar com o prompt padr�o ou sair
         # exit(1) # Descomente para sair se o prompt for essencial

    main_loop = None
    try:
        logger.info(f"Iniciando Trackie no modo: {args.mode}")
        # Cria a inst�ncia principal
        main_loop = AudioLoop(video_mode=args.mode, show_preview=show_actual_preview)
        # Executa o loop principal ass�ncrono
        asyncio.run(main_loop.run())

    except KeyboardInterrupt:
        logger.info("\nInterrup��o pelo teclado recebida (Ctrl+C). Encerrando...")
        if main_loop:
            logger.info("Sinalizando parada para as tarefas...")
            main_loop.stop_event.set()
            # D� um pequeno tempo para as tarefas tentarem limpar antes de sair
            # time.sleep(1) # Opcional
    except Exception as e:
        # Captura qualquer outra exce��o n�o tratada no n�vel superior
        logger.error(f"Erro inesperado e n�o tratado no bloco __main__: {type(e).__name__}: {e}")
        traceback.print_exc()
        if main_loop:
            logger.info("Sinalizando parada devido a erro inesperado...")
            main_loop.stop_event.set()
    finally:
        # Este bloco sempre ser� executado, mesmo ap�s interrup��o ou erro
        logger.info("Bloco __main__ finalizado.")
        # Verifica se PyAudio ainda precisa ser terminado (caso run() n�o tenha chegado ao fim)
        # if pya and main_loop and not main_loop.stop_event.is_set(): # Se run n�o terminou normalmente
        #      try:
        #          logger.info("Tentando terminar PyAudio no finally do main...")
        #          pya.terminate()
        #      except Exception as e_pya_final:
        #          logger.error(f"Erro ao terminar PyAudio no finally do main: {e_pya_final}")

        logger.info("Programa completamente finalizado.")