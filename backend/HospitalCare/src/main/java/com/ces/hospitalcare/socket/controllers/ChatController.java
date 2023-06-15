package com.ces.hospitalcare.socket.controllers;
import com.ces.hospitalcare.socket.model.Message;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Controller;

@Controller
public class ChatController {
  @Autowired
  private SimpMessagingTemplate simpMessagingTemplate;

  @MessageMapping("/private-message")
  public Message receiveMessage(@Payload Message message) {
    simpMessagingTemplate.convertAndSendToUser(message.getReceiverName(), "/private", message);
//    System.out.println(message.toString());
    return message;
  }
}