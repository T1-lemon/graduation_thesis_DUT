export const EMAIL_REGEX = /^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$/
export const NAME_REGEX = /^[a-zA-Z]+(?:[ ]{1}[a-zA-Z]+)*$/
export const PASSWORD_REGEX =
  /^(?=.*[a-zA-Z])(?=.*\d)(?=.*[!@#$%^&*()_+])[A-Za-z\d][A-Za-z\d!@#$%^&*()_+]{7,19}$/

export const PHONE_REGEX_VN =
  /^(0|84)(2[03]|3[25]|4[12567]|5[2689]|6[2389]|7[06789]|8[15689]|9[0356789])\d{7}$/
