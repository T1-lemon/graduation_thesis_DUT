package com.ces.hospitalcare.service;
import com.ces.hospitalcare.dto.MessageDTO;
import com.ces.hospitalcare.http.request.MessageRequest;
import com.ces.hospitalcare.http.response.MessageResponse;
import java.util.List;

public interface IMessageService {
  List<MessageResponse> getAllMessageTwoUser(Long senderId, Long receiverId);

  MessageDTO addMessage(MessageRequest messageRequest);
}
