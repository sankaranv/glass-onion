import transformers
from transformers.utils import ModelOutput
import torch


class GPT2Config(transformers.GPT2Config):
    def __init__(self, *args, **kwargs):
        """
        This doesn't do anything except wrap the transformers.GPT2Config class
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)


class GPT2Tokenizer(transformers.GPT2Tokenizer):
    def __init__(self, *args, **kwargs):
        """
        This doesn't do anything except wrap the transformers.GPT2Tokenizer class
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)


class GPT2LMModel(transformers.GPT2LMHeadModel):
    """
    A GPT2 model that can be intervened on.
    """

    def __init__(self, config: GPT2Config):
        super().__init__(config)

    def apply_layer_intervention(
            self,
            hidden_states,
            intervene_at_position,
            intervened_value=0,
    ):
        """
        Applies the attention intervention to a given layer, head, and token.
        """

        new_states = hidden_states.clone()
        new_states[:, intervene_at_position, :] = intervened_value
        return new_states

    def apply_attn_head_intervention(
            self,
            attn_module,
            hidden_states,
            intervene_at_position,
            intervene_at_head,
            intervened_value=1e-20,
    ):
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        attention_scores = attn_module(hidden_states)
        attention_scores[0][0, intervene_at_head, :] = intervened_value
        return attention_scores

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            intervene_at_layer=None,
            intervene_attn_head=None,
            intervened_value=0,
            intervene_at_position=0,
    ):
        # Perform the full forward pass
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # Copy the hidden states at the intervened layer
        # This is the default value with no intervention applied
        hidden_states = outputs.hidden_states
        attention_scores = outputs.attentions

        if intervene_at_layer is not None:
            layer_block = self.transformer.h[intervene_at_layer]
            layer_hidden_states = hidden_states[intervene_at_layer]
            if intervene_attn_head is None:
                # Apply layer intervention
                hidden_states = self.apply_layer_intervention(
                    layer_hidden_states,
                    intervene_at_position,
                    intervened_value,
                )
            else:
                # Apply attention head intervention
                attention_output = self.apply_attn_head_intervention(
                    layer_block.attn,
                    hidden_states,
                    intervene_at_position,
                    intervene_attn_head,
                    intervened_value,
                )
                hidden_states = layer_block.mlp(layer_block.ln_2(attention_output))

                # Apply layer normalization after attention block
                hidden_states = layer_block.ln_1(hidden_states)

            # Continue forward with the remaining layers
            for i in range(intervene_at_layer + 1, self.config.num_hidden_layers):
                layer_block = self.transformer.h[i]
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]

                # Forward pass through the layer
                hidden_states = layer_block(hidden_states)

        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        logits = self.lm_head(hidden_states)

        # Construct a ModelOutput object
        model_output = ModelOutput(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        return model_output
