package com.ces.hospitalcare.dto;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class ConversationDTO extends BaseDTO {
//  private UserDTO sender;

  private UserDTO receiver;
}
