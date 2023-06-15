import CircularProgress from '@mui/material/CircularProgress'
import React, { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { verifyEmailTokenService } from '../../../services/authService'
import { checkResponseSuccess } from '../../../utils/checkResponseStatus'

type Props = {}
const verifySuccess = require('../../../assets/img/verify_success.png')
const verifyFailed = require('../../../assets/img/verify_failed.png')

const RegisterVerifyStatus = (props: Props) => {
  const params = useParams()
  const [isVerify, setIsVerify] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    if (params) {
      const fetchApiVerifyEmailToken = async () => {
        setIsLoading(true)
        const response: any = await verifyEmailTokenService(
          params.token as string
        )
        setIsLoading(false)
        if (checkResponseSuccess(response.status)) {
          setIsVerify(true)
        }
        console.log(response)
      }
      fetchApiVerifyEmailToken()
    }
  }, [])

  return (
    <div className='verify__div--container'>
      {isLoading ? (
        <div className='verify__loading'>
          <div className='verify__progress'>
            <CircularProgress
              color='inherit'
              sx={{ width: '100px', height: '100px' }}
            />
          </div>
          <h1 className='verify__h1'>
            Please wait a few minutes, your code is being verified.
          </h1>
        </div>
      ) : isVerify ? (
        <div className='verify__div--success'>
          <img className='verify__img--email' src={verifySuccess} alt='' />
          <h1 className='verify__h1'>
            Congratulations, your account has been verified.
          </h1>
          <Link to='/login' className='verify__link'>
            Click here to login
          </Link>
        </div>
      ) : (
        <div className='verify__div--success'>
          <img className='verify__img--email' src={verifyFailed} alt='' />
          <h1 className='verify__h1'>Sorry, We could not verify account.</h1>
          <h1 className='verify__h1'>
            It maybe already verified, or verification cod is incorrect.
          </h1>
        </div>
      )}
    </div>
  )
}

export default RegisterVerifyStatus
