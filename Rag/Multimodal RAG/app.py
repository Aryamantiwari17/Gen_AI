#ollama create codeguru -f modelfile
import requests
import json
import gradio as gr

url="http://localhost"#got from document of ollama

header={
    'contetn-type':"apllication/json"
}

history=[]

def genrate_response(prompt):
    history.append(prompt)
    final_prompt='/n'.join(history)
    
    data={
        "model":"codeguru",
        "prompt":final_prompt,
        "stream":False
         }
    
    response=requests.post(url,headers=header,data=json.dumps(data))
    
    if response.status_code==200:
        response=response.text
        data=json.loads(response)
        actual_response=data['response']
        return actual_response
    
    else:
        print("error",response.text)
    
    #frontend
    interface=interface(
        fn=genrate_response,
        inputs=gr.Textbox(lines=4,placeholder="Enter your prompt"),
        output="text"
    )
    
    interface.launch()
