package com.ces.hospitalcare.dto;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class MessageDTO extends BaseDTO {
  private String text;

  private ConversationDTO conversation;
}
