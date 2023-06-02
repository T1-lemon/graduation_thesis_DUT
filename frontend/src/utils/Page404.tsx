import { useNavigate } from 'react-router-dom'

import '../assets/css/components/page_404.css'
import ButtonCustomize from '../components/ButtonCustomize'

const logo = require('../assets/img/robot-404.png')

function Page404() {
  const navigate = useNavigate()
  const onClickBtn = () => {
    navigate('/home')
  }

  return (
    <>
      <div className='box__404'>
        <div className='box__404--text'>
          <div className='text--404'>404</div>
          <div className='text--title'>
            Ooops! <span>Page Not Found</span>
          </div>
          <div className='text--notfound'>
            This page doesn't exist or was removed!
            <span>We suggest you back to home</span>
          </div>
          <div>
            <ButtonCustomize
              text='Back to Home'
              className='btn__radius'
              onClickBtn={onClickBtn}
            />
          </div>
        </div>
        <div className='box__404--img'>
          <img className='robot__404--img' src={logo} alt='' />
        </div>
      </div>
    </>
  )
}

export default Page404
