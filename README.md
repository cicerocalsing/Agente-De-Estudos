# 🧠 Agente de Estudos

Um agente inteligente para análise e estudo de documentos PDF usando IA.

## 🚀 Funcionalidades

- **Upload de PDFs** - Carrega e processa documentos
- **Análise inteligente** - Responde perguntas sobre o conteúdo
- **Classificação automática** - Detecta o tipo de pergunta
- **Memória persistente** - Salva conversas para referência futura
- **Interface web** - Interface amigável com Streamlit

## 📁 Estrutura do Projeto

```
Agente-De-Estudos/
├── main_simple.py              # Aplicação principal
├── config.py                   # Configurações
├── memory_manager.py           # Gerenciamento de memória
├── document_processor_simple.py # Processamento de PDFs
├── llm_service_simple.py       # Serviço de IA
├── esqueleto_do_projeto.txt    # Documentação do projeto
├── proximos_passos.txt         # Próximos passos
└── temp_pdf/                   # PDFs temporários
```

## 🛠️ Instalação

1. **Instale as dependências principais:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configurar variáveis de ambiente:**
   - `HF_TOKEN`: Token do HuggingFace
   - `MEM0_API_KEY`: Chave da API Mem0

3. **Executar a aplicação:**
```bash
streamlit run main_simple.py
```

## 🎯 Como Usar

1. **Carregue um PDF** na interface
2. **Digite sua pergunta** sobre o documento
3. **Clique em "Executar"** para obter a resposta
4. **Visualize a memória** para ver conversas anteriores

## 🔧 Tecnologias

- **Streamlit** - Interface web
- **PyPDF2** - Processamento de PDFs
- **Mem0** - Sistema de memória
- **HuggingFace** - Modelos de IA (com fallback)

## 📝 Arquivos de Documentação

- `esqueleto_do_projeto.txt` - Estrutura inicial do projeto
- `proximos_passos.txt` - Sugestões para melhorias futuras

## 🎉 Status

✅ **Funcional** - Sistema completo e operacional
✅ **Robusto** - Fallbacks para garantir funcionamento
✅ **Limpo** - Código organizado e documentado 