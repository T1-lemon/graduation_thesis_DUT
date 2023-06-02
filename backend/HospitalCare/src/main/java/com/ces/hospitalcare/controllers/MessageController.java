package com.ces.hospitalcare.controllers;
import com.ces.hospitalcare.dto.MessageDTO;
import com.ces.hospitalcare.dto.UserDTO;
import com.ces.hospitalcare.http.request.MessageRequest;
import com.ces.hospitalcare.http.response.MessageResponse;
import com.ces.hospitalcare.http.response.UserResponse;
import com.ces.hospitalcare.service.IMessageService;
import com.ces.hospitalcare.service.IUserService;
import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/messages")
public class MessageController {
  @Autowired
  private IMessageService messageService;

  @Autowired
  private IUserService userService;

  @GetMapping(value = "")
  @PreAuthorize("hasAnyAuthority('ROLE_DOCTOR', 'ROLE_PATIENT')")
  public List<MessageResponse> getAllMessageTwoUser(
      @RequestParam("receiverId") Long receiverId) {
    UserResponse userResponse = userService.getCurrentUser();
    UserDTO sender = userResponse.getUser();

    return messageService.getAllMessageTwoUser(sender.getId(), receiverId);
  }

  @PostMapping(value = "")
  @PreAuthorize("hasAnyAuthority('ROLE_DOCTOR', 'ROLE_PATIENT')")
  public MessageDTO addMessage(@RequestBody MessageRequest messageRequest) {
    UserResponse userResponse = userService.getCurrentUser();
    UserDTO sender = userResponse.getUser();

    return messageService.addMessage(messageRequest);
  }
}
