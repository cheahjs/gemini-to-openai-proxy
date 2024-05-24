package openai

type EmbedRequest struct {
	Input          interface{} `json:"input"`
	Model          string      `json:"model"`
	EncodingFormat string      `json:"encoding_format,omitempty"`
	Dimensions     int         `json:"dimensions,omitempty"`
	User           string      `json:"user,omitempty"`
}

type EmbedResponseData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

type Usage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type EmbedResponse struct {
	Object string               `json:"object"`
	Data   []*EmbedResponseData `json:"data"`
	Model  string               `json:"model"`
	Usage  *Usage               `json:"usage"`
}

type ModelResponse struct {
	Object string               `json:"object"`
	Data   []*ModelResponseData `json:"data"`
}

type ModelResponseData struct {
	Object  string `json:"object"`
	ID      string `json:"id"`
	Created uint   `json:"created"`
	OwnedBy string `json:"owned_by"`
}
