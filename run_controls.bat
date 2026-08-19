@echo off
cd /d "%~dp0"
set PY=C:\Python314\python.exe
set TXT=D:\paper_corpus_sinq.txt
%PY% flip_diagnosis.py --model "D:/models/Qwen3.5-4B" --bits 4 --groupsize 128 --seqlen 512 --n_samples 6 --dataset %TXT% --topk 8 --scope all --chunk 1 --device cpu > result_all_b4.txt 2>&1
%PY% flip_diagnosis.py --model "D:/models/Qwen3.5-4B" --bits 3 --groupsize 128 --seqlen 512 --n_samples 6 --dataset %TXT% --topk 8 --scope head --chunk 1 --device cpu > result_head_b3.txt 2>&1
%PY% flip_diagnosis.py --model "D:/models/Qwen3.5-4B" --bits 3 --groupsize 128 --seqlen 512 --n_samples 6 --dataset %TXT% --topk 8 --scope body --chunk 1 --device cpu > result_body_b3.txt 2>&1
echo DONE > _controls_done.marker
