Detecção de Pessoas e Quedas com Visão Computacional

Olá professor! Este é o readme do projeto. A ideia é detectar quedas. Imagino que possa ser usado, por exemplo. em hospitais, ou em casas com pessoas idosas. Quando o Sr rodar, vai perceber que o vídeo incluso tem falsos positivos e, às vezes, não detecta quedas. Podia ter enviado um vídeo mais certeiro, onde todas as quedas são registradas corretamente. No entanto, acho importante mostrar as fraquezas do projeto também. Obrigado!

Para rodar o projeto, basta apenas executar o comando "python main.py"

Funcionalidades

1. Detecção de Pessoas
   - Usa o modelo SSD MobileNet V2 para detectar pessoas em vídeos.
   - Desenha caixas verdes ao redor das pessoas detectadas.

2. Análise de Poses
   - Utiliza o MediaPipe Pose para identificar partes do corpo.

3. Identificação de Quedas
   - Detecta quedas com base na posição da cabeça ou quadril em relação ao chão.
   - Troca a cor da caixa para laranja se identificar uma queda.
   - Exibe "Queda Detectada!" na tela por 2 segundos.

4. Contador de Quedas com Cooldown
   - Evita que o contador dispare infinitamente para a mesma pessoa.
   - Cada pessoa só pode registrar uma queda a cada 1 segundo.

Estrutura do Projeto

Arquivos e Pastas

- `main.py`:
  - Código principal do projeto.

- `frozen_inference_graph.pb`:
  - Pesos do modelo SSD MobileNet V2.

- `ssd_mobilenet_v2_coco.pbtxt`:
  - Configuração do modelo SSD MobileNet V2.

- `quedas.mp4`:
  - Um vídeo de exemplo para testar o programa.


