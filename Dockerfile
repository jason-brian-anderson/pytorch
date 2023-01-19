FROM pytorch/pytorch

LABEL name="pytorch_jupyter"

RUN pip install -r requirements.txt

EXPOSE 80

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=80", "--no-browser", "--allow-root", "--NotebookApp.token=''"]