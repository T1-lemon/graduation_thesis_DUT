package com.ces.hospitalcare.service.impl;
import com.ces.hospitalcare.dto.MessageDTO;
import com.ces.hospitalcare.entity.MessageEntity;
import com.ces.hospitalcare.http.request.MessageRequest;
import com.ces.hospitalcare.http.response.MessageResponse;
import com.ces.hospitalcare.repository.ConversationRepository;
import com.ces.hospitalcare.repository.MessageRepository;
import com.ces.hospitalcare.service.IMessageService;
import java.util.ArrayList;
import java.util.List;
import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class MessageServiceImpl implements IMessageService {
  @Autowired
  private MessageRepository messageRepository;

  @Autowired
  private ConversationRepository conversationRepository;

  @Autowired
  private ModelMapper mapper;

  @Override
  public List<MessageResponse> getAllMessageTwoUser(Long senderId, Long receiverId) {
    List<MessageEntity> messageEntityList = messageRepository.getAllMessageTwoUser(senderId,
        receiverId);
    List<MessageResponse> messageResponseList = new ArrayList<>();
    for (MessageEntity messageEntity : messageEntityList) {
      Long senderMessageId = messageEntity.getConversation().getSender().getId();
      MessageResponse messageResponse = MessageResponse.builder().message(messageEntity.getText())
          .fromSelf(senderMessageId == senderId).build();
      messageResponseList.add(messageResponse);
    }

    return messageResponseList;
  }

  @Override
  public MessageDTO addMessage(MessageRequest messageRequest) {
    MessageEntity messageEntity = MessageEntity.builder().text(messageRequest.getText())
        .conversation(conversationRepository.getReferenceById(messageRequest.getConversationId()))
        .build();
    messageRepository.save(messageEntity);
    return mapper.map(messageEntity, MessageDTO.class);
  }
}
