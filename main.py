import cv2
import numpy as np
import mediapipe as mp
import time

ARQUIVO_VIDEO = 'quedas.mp4'
ARQUIVO_MODELO = 'frozen_inference_graph.pb'
ARQUIVO_CFG = 'ssd_mobilenet_v2_coco.pbtxt'

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

contador_quedas = 0
tempo_aviso = 0  # Temporizador para o aviso de queda - Aviso de queda é uma mensagem que aparece na tela quando ocorre uma queda
duracao_aviso = 2  # Duração mínima do aviso
cooldown_queda = 1  # Tempo mínimo entre registros de quedas - O objetivo é evitar que quedas em excesso sejam registradas
ultimo_registro = {}  # Dicionário para armazenar o último tempo de queda por pessoa


def carregar_modelo(ARQUIVO_MODELO, ARQUIVO_CFG):
    try:
        modelo = cv2.dnn.readNetFromTensorflow(ARQUIVO_MODELO, ARQUIVO_CFG)
    except cv2.error as erro:
        print(f"Erro ao carregar o modelo: {erro}")
        exit()
    return modelo


def aplicar_supressao_nao_maxima(caixas, confiancas, limiar_conf, limiar_supr):
    indices = cv2.dnn.NMSBoxes(caixas, confiancas, limiar_conf, limiar_supr)
    return [caixas[i] for i in indices.flatten()] if len(indices) > 0 else []


def verificar_queda(frame, caixas_finais):
    """
    Processa a pose apenas dentro das caixas delimitadoras detectadas
    e registra quedas com cooldown.
    """
    global contador_quedas, tempo_aviso, ultimo_registro
    altura, largura, _ = frame.shape
    estado_queda = {}
    queda_detectada = False

    for id_pessoa, (inicioX, inicioY, largura_caixa, altura_caixa) in enumerate(caixas_finais):
        # Garantir que as coordenadas estejam dentro dos limites do frame
        inicioX = max(0, inicioX)
        inicioY = max(0, inicioY)
        fimX = min(largura, inicioX + largura_caixa)
        fimY = min(altura, inicioY + altura_caixa)

        # Recortar a região de interesse da caixa delimitadora
        roi = frame[inicioY:fimY, inicioX:fimX]

        # Verificar se a ROI não está vazia
        if roi.size == 0:
            estado_queda[id_pessoa] = False
            continue

        # Converter ROI para RGB e processar com MediaPipe
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        resultados = pose.process(roi_rgb)

        if resultados.pose_landmarks:
            # Filtrar landmarks que estão fora da ROI ou são irreais
            landmarks_validos = []
            for landmark in resultados.pose_landmarks.landmark:
                x = int(inicioX + landmark.x * largura_caixa)
                y = int(inicioY + landmark.y * altura_caixa)

                # Verifique se o ponto está dentro da caixa delimitadora
                if inicioX <= x <= fimX and inicioY <= y <= fimY:
                    landmarks_validos.append((x, y))
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            # Se não houver landmarks válidos, ignore a pessoa
            if not landmarks_validos:
                estado_queda[id_pessoa] = False
                continue

            # Verificar se há queda (cabeça ou quadril baixos no frame completo)
            cabeca = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            quadril = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            cabeca_perto_chao = (inicioY + cabeca.y * altura_caixa) / altura > 0.8
            quadril_perto_chao = (inicioY + quadril.y * altura_caixa) / altura > 0.8

            # Verifica se a queda é válida e respeita o cooldown
            tempo_atual = time.time()
            if (cabeca_perto_chao or quadril_perto_chao) and \
                    (id_pessoa not in ultimo_registro or tempo_atual - ultimo_registro[id_pessoa] > cooldown_queda):
                contador_quedas += 1
                ultimo_registro[id_pessoa] = tempo_atual  # Atualiza o tempo do último registro
                tempo_aviso = tempo_atual  # Reinicia o temporizador do aviso
                estado_queda[id_pessoa] = True
                queda_detectada = True
            else:
                estado_queda[id_pessoa] = False
        else:
            estado_queda[id_pessoa] = False

    return frame, estado_queda, queda_detectada


def main():
    global contador_quedas, tempo_aviso
    captura = cv2.VideoCapture(ARQUIVO_VIDEO)
    detector_pessoas = carregar_modelo(ARQUIVO_MODELO, ARQUIVO_CFG)
    pausado = False

    while True:
        if not pausado:
            ret, frame = captura.read()
            if not ret:
                break

            # Criação do blob a partir do frame e realização da detecção
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
            detector_pessoas.setInput(blob)
            deteccoes = detector_pessoas.forward()

            caixas = []
            confiancas = []

            # Extração das caixas delimitadoras e confianças das detecções
            for i in range(deteccoes.shape[2]):
                confianca = deteccoes[0, 0, i, 2]
                if confianca > 0.3:  # Limiar reduzido para aumentar a sensibilidade
                    (altura, largura) = frame.shape[:2]
                    caixa = deteccoes[0, 0, i, 3:7] * np.array([largura, altura, largura, altura])
                    (inicioX, inicioY, fimX, fimY) = caixa.astype("int")
                    caixas.append([inicioX, inicioY, fimX - inicioX, fimY - inicioY])
                    confiancas.append(float(confianca))

            # Aplicação da supressão não máxima para finalizar as caixas delimitadoras
            caixas_finais = aplicar_supressao_nao_maxima(caixas, confiancas, limiar_conf=0.3, limiar_supr=0.4)
            numero_pessoas = len(caixas_finais)

            # Processar a pose com MediaPipe e verificar quedas
            frame, estado_queda, queda_detectada = verificar_queda(frame, caixas_finais)

            # Desenho das caixas e exibição do número de pessoas detectadas
            for id_pessoa, (inicioX, inicioY, largura, altura) in enumerate(caixas_finais):
                # Alterar a cor da caixa se houver uma queda
                cor = (0, 255, 0)  # Verde padrão
                if estado_queda.get(id_pessoa, False):
                    cor = (0, 165, 255)  # Laranja (em BGR)

                cv2.rectangle(frame, (inicioX, inicioY), (inicioX + largura, inicioY + altura), cor, 2)

            cv2.putText(frame, f"Pessoas: {numero_pessoas}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Exibir contador de quedas no frame
            cv2.putText(frame, f"Quedas: {contador_quedas}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Exibir mensagem de aviso se houver queda recente
            if time.time() - tempo_aviso < duracao_aviso:
                cv2.putText(frame, "Queda Detectada!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Exibição do frame processado e controle de pausa/play
        cv2.imshow("Rastreio de Pessoas e Poses", frame)

        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('p'):
            pausado = not pausado

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


def main():
    global contador_quedas, tempo_aviso
    captura = cv2.VideoCapture(ARQUIVO_VIDEO)
    detector_pessoas = carregar_modelo(ARQUIVO_MODELO, ARQUIVO_CFG)
    pausado = False

    while True:
        if not pausado:
            ret, frame = captura.read()
            if not ret:
                break

            # Criação do blob a partir do frame e realização da detecção
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
            detector_pessoas.setInput(blob)
            deteccoes = detector_pessoas.forward()

            caixas = []
            confiancas = []

            # Extração das caixas delimitadoras e confianças das detecções
            for i in range(deteccoes.shape[2]):
                confianca = deteccoes[0, 0, i, 2]
                if confianca > 0.3:  # Limiar reduzido para aumentar a sensibilidade
                    (altura, largura) = frame.shape[:2]
                    caixa = deteccoes[0, 0, i, 3:7] * np.array([largura, altura, largura, altura])
                    (inicioX, inicioY, fimX, fimY) = caixa.astype("int")
                    caixas.append([inicioX, inicioY, fimX - inicioX, fimY - inicioY])
                    confiancas.append(float(confianca))

            # Aplicação da supressão não máxima para finalizar as caixas delimitadoras
            caixas_finais = aplicar_supressao_nao_maxima(caixas, confiancas, limiar_conf=0.3, limiar_supr=0.4)
            numero_pessoas = len(caixas_finais)

            # Desenho das caixas e exibição do número de pessoas detectadas
            for (inicioX, inicioY, largura, altura) in caixas_finais:
                cv2.rectangle(frame, (inicioX, inicioY), (inicioX + largura, inicioY + altura), (0, 255, 0), 2)
            cv2.putText(frame, f"Pessoas: {numero_pessoas}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Processar a pose com MediaPipe e verificar quedas
            frame, queda_detectada = verificar_queda(frame, caixas_finais)

            # Exibir contador de quedas no frame
            cv2.putText(frame, f"Quedas: {contador_quedas}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Exibir mensagem de aviso se houver queda recente
            if time.time() - tempo_aviso < duracao_aviso:
                cv2.putText(frame, "Queda Detectada!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Exibição do frame processado e controle de pausa/play
        cv2.imshow("Rastreio de Pessoas e Poses", frame)

        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('p'):
            pausado = not pausado

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
