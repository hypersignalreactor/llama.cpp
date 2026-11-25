import subprocess, json, re, os, sys
P=["In the beginning God created the heaven and the",
"The LORD is my shepherd; I shall not want. He maketh me to",
"For God so loved the world, that he gave his only begotten Son,",
"Our Father which art in heaven, Hallowed be thy name. Thy kingdom",
"Blessed are the poor in spirit: for theirs is the kingdom of heaven. Blessed are they that",
"In the beginning was the Word, and the Word was with God, and",
"Question: What did John the Baptist eat?\nAnswer:",
"Question: Who were the first disciples Jesus called by the Sea of Galilee?\nAnswer:",
"Question: What did Jesus say to the paralytic in Capernaum?\nAnswer:",
"Problem: A shepherd has 100 sheep. He sells 37 of them and then buys 12 more. How many sheep does he have now?\nSolution:",
"The four Gospels of the New Testament are",
"Photosynthesis is the process by which"]
C=[('full',{}),('p0.90',{'OURO_EXIT_P':'0.9'}),('p0.95',{'OURO_EXIT_P':'0.95'}),('p0.99',{'OURO_EXIT_P':'0.99'}),('p0.90 min2',{'OURO_EXIT_P':'0.9','OURO_MIN_LOOPS':'2'}),('p0.95 min2',{'OURO_EXIT_P':'0.95','OURO_MIN_LOOPS':'2'}),('p0.99 min2',{'OURO_EXIT_P':'0.99','OURO_MIN_LOOPS':'2'})]
out={}
for name,env in C:
    R=[]
    for p in P:
        e=dict(os.environ); e.update(env)
        r=subprocess.run(['./llama.cpp/build/bin/llama-completion','-m','hf/Ouro-1.4B-Q8_0-gate.gguf','-c','1024','-p',p,'-n','40','--temp','0','-no-cnv','-t','2'],capture_output=True,text=True,env=e)
        m=re.search(r' eval time =\s+([\d.]+) ms /\s+(\d+) runs',r.stderr); a=re.search(r'(\d+) tokens, ([\d.]+) loops on average',r.stderr)
        R.append({'text':r.stdout[len(p):],'ms':float(m.group(1)),'n':int(m.group(2)),'loops':float(a.group(2)) if a else 4.0})
    out[name]=R; json.dump(out,open('gen_trial.json','w'),indent=1); print(name,'done',flush=True)
