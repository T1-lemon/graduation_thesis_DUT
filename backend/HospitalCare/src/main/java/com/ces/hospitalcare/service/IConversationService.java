package com.ces.hospitalcare.service;
import com.ces.hospitalcare.dto.ConversationDTO;
import java.util.List;

public interface IConversationService {
  List<ConversationDTO> getAllConversationOfSender(Long senderId);

  ConversationDTO addConversation(Long senderId, Long receiverId);
}
