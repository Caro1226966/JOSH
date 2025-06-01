# stream = ollama.chat(
#                         model=MODEL,
#                         messages=[{'role': 'user', 'content': command + ' in less than 30 words'}],
#                         stream=True
#                     )
#                     answer = ''
#                     for chunk in stream:
#                         answer += chunk['message']['content']
#                     print(answer)
#                     pyttsx3.speak(answer)