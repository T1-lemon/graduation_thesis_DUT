import React from 'react'

import '../../../assets/css/pages/guestPage/authPage/register_verify_email.css'

type Props = {}
const verifyEmail = require('../../../assets/img/verify_email.png')

const RegisterVerifyEmail = (props: Props) => {
  return (
    <div className='verify__div--container'>
      <img className='verify__img--email' src={verifyEmail} alt='' />
      <h1 className='verify__h1'>You have signed up successfully!</h1>
      <p className='verify__p'>Please check your email to verify account</p>
    </div>
  )
}

export default RegisterVerifyEmail
