package openai

import (
	"github.com/google/generative-ai-go/genai"
	"github.com/pkg/errors"
)

func ConvertOpenAIRequestToGemini(openAIReq *EmbedRequest, model *genai.EmbeddingModel) (*genai.EmbeddingBatch, error) {
	if openAIReq.EncodingFormat != "" && openAIReq.EncodingFormat != "float" {
		return nil, errors.New("unsupported encoding format")
	}

	geminiBatchReq := model.NewBatch()
	switch v := openAIReq.Input.(type) {
	case string:
		geminiBatchReq.AddContent(genai.Text(v))
	case []interface{}:
		for _, text := range v {
			if t, ok := text.(string); ok {
				geminiBatchReq.AddContent(genai.Text(t))
			} else {
				return nil, errors.Errorf("unsupported input type: %T", t)
			}
		}
	default:
		return nil, errors.Errorf("unsupported input type: %T", v)
	}

	return geminiBatchReq, nil
}

func ConvertGeminiResponseToOpenAI(geminiBatchResp *genai.BatchEmbedContentsResponse, model string) *EmbedResponse {
	openAIResp := &EmbedResponse{
		Object: "list",
		Model:  model,
	}

	for i, geminiResp := range geminiBatchResp.Embeddings {
		openAIResp.Data = append(openAIResp.Data, &EmbedResponseData{
			Object:    "embedding",
			Embedding: geminiResp.Values,
			Index:     i,
		})
	}

	openAIResp.Usage = &Usage{
		PromptTokens: 0,
		TotalTokens:  0,
	}

	return openAIResp
}
