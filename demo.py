
import re
import torch
import argparse
from tqdm import tqdm
from model import Predictor, NewSE2, Stalking
from transformers import AutoModel, AutoTokenizer


class Demo(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            "스토킹 위험도 지수 출력 Demo"
        )
        self.parse_args()
        self.arglist = self.parser.parse_args()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 텍스트 임베딩
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
        self.embed_model = AutoModel.from_pretrained("beomi/KcELECTRA-base-v2022")
        self.embed_model = self.embed_model.to(device=self.device)
        self.embed_model.eval()

        # 정신건강 분류 모델
        self.mentality_model = NewSE2(num_layers=2,
                                      embedding_size=768,
                                      hidden_size=384,
                                      attention_dim=512,
                                      keys=30,
                                      fc_dim=512,
                                      batch_size=25)

        # 스토킹 잠재/가해 분류 모델
        self.stalking_model = Stalking(device=self.device, model_path=self.arglist.SE_path, batch_size=64)

    def parse_default_args(self):
        self.parser.add_argument("--input_text", type=str)  # 테스트하고자 하는 텍스트(.txt) 한 줄당 1개의 설문조사 문항 답변
        self.parser.add_argument("--SE_path", default='./TRAINED_MODEL/SE', type=str)
        self.parser.add_argument("--mental_model", default='./TRAINED_MODEL/MENTAL/mentality_test.pt', type=str)
        self.parser.add_argument("--stalking_model", default='./TRAINED_MODEL/Classifier/stalking_detection.pt',
                                 type=str)

    def parse_args(self):
        self.parse_default_args()

    def make_input(self):
        # 테스트용 설문조사 데이터
        with open(self.arglist.input_text, "r") as f:
            survey = f.readlines()

        # 정규표현식
        for i in range(len(survey)):
            survey[i] = re.sub(f'[^A-Za-z가-힣 ]', '', survey[i])

        # 토큰화 + KcELECTRA 임베딩
        temp = []

        for i in tqdm(range(len(survey))):
            tokenized = self.tokenizer.encode_plus(survey[i], max_length=512, padding='max_length', truncation=True,
                                                   return_tensors='pt')
            tokenized = tokenized.to(device=self.device)

            with torch.no_grad():
                embedded = self.embed_model(input_ids=tokenized['input_ids'],
                                            token_type_ids=tokenized['token_type_ids'],
                                            attention_mask=tokenized['attention_mask'])
                temp.append(embedded[0].squeeze(0))

            del tokenized

        self.result = {
            'last': temp[0].unsqueeze(0).to(device=self.device, dtype=torch.float32),
            'reason': temp[1].unsqueeze(0).to(device=self.device, dtype=torch.float32),
            'action': temp[2].unsqueeze(0).to(device=self.device, dtype=torch.float32),
            'try_': temp[3].unsqueeze(0).to(device=self.device, dtype=torch.float32),
            'reaction': temp[4].unsqueeze(0).to(device=self.device, dtype=torch.float32),
            'valuable': temp[5].unsqueeze(0).to(device=self.device, dtype=torch.float32),
            'start': temp[6].unsqueeze(0).to(device=self.device, dtype=torch.float32),
            'charmingLover': temp[7].unsqueeze(0).to(device=self.device, dtype=torch.float32),
            'charmingCustomer': temp[8].unsqueeze(0).to(device=self.device, dtype=torch.float32),
            'relation': temp[9].unsqueeze(0).to(device=self.device, dtype=torch.float32),
            'event': temp[10].unsqueeze(0).to(device=self.device, dtype=torch.float32)
        }

    def ready_model(self):
        self.mentality_model.load_state_dict(torch.load(self.arglist.mental_model, map_location=self.device))
        self.stalking_model.load_state_dict(torch.load(self.arglist.stalking_model, map_location=self.device))

        # 스토킹 위험도 지수 예측 모델 불러오기
        self.pred = Predictor(mentality_model=self.mentality_model,
                              stalking_model=self.stalking_model,
                              batch_size=1)

        return self.pred

    def prediction(self):
        print("\n1. 설문조사 답변 변환 중...\n")
        self.make_input()
        print("\n2. 스토킹 위험도 지수 예측 모델 구성 중...\n")
        self.ready_model()
        print("\n3. 스토킹 위험도 지수 예측 진행 중...\n")
        result = self.pred.predict(self.result['last'],
                                   self.result['reason'],
                                   self.result['action'],
                                   self.result['try_'],
                                   self.result['reaction'],
                                   self.result['charmingCustomer'],
                                   self.result['relation'])

        attention_by_question = self.pred.stalking_model.att_fin

        print("\n\n\n4. 스토킹 위험도 지수 예측 결과 확인!\n")
        print(f"   1) 스토킹 위험도 지수: {result.squeeze(1).cpu().detach().numpy().tolist()[0]:5.4f}")
        print(f"   2) 문항별 중요도 분석:")
        print(f"      - 1번 문항  : {attention_by_question.squeeze(1).cpu().detach().numpy().tolist()[0][0] * 100:4.2f}%")
        print(f"      - 2번 문항  : {attention_by_question.squeeze(1).cpu().detach().numpy().tolist()[0][1] * 100:4.2f}%")
        print(f"      - 3번 문항  : {attention_by_question.squeeze(1).cpu().detach().numpy().tolist()[0][2] * 100:4.2f}%")
        print(f"      - 4번 문항  : {attention_by_question.squeeze(1).cpu().detach().numpy().tolist()[0][3] * 100:4.2f}%")
        print(f"      - 5번 문항  : {attention_by_question.squeeze(1).cpu().detach().numpy().tolist()[0][4] * 100:4.2f}%")
        print(f"      - 10번 문항 : {attention_by_question.squeeze(1).cpu().detach().numpy().tolist()[0][5] * 100:4.2f}%")


if __name__ == '__main__':
    demo = Demo()
    demo.prediction()
