package com.ces.hospitalcare.controllers;
import com.ces.hospitalcare.dto.ConversationDTO;
import com.ces.hospitalcare.dto.UserDTO;
import com.ces.hospitalcare.http.response.UserResponse;
import com.ces.hospitalcare.service.IConversationService;
import com.ces.hospitalcare.service.IUserService;
import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/conversations")
public class ConversationController {
  @Autowired
  private IConversationService conversationService;

  @Autowired
  private IUserService userService;

  @GetMapping(value = "/receivers")
  @PreAuthorize("hasAnyAuthority('ROLE_DOCTOR', 'ROLE_PATIENT')")
  public List<ConversationDTO> getAllConversationBySender() {
    UserResponse userResponse = userService.getCurrentUser();
    UserDTO sender = userResponse.getUser();

    return conversationService.getAllConversationOfSender(sender.getId());
  }
}
