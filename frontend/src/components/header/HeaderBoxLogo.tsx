import { Box } from '@mui/material'
import { Link } from 'react-router-dom'

const logo = require('../../assets/img/logo-hospital-care.png')

function HeaderBoxLogo() {
  return (
    <Box className='header__box--logo'>
      <img className='header__img--logo' src={logo} alt='' />
      <Link to='/home' className='header__title--slogan'>
        Hospital Care
      </Link>
    </Box>
  )
}

export default HeaderBoxLogo
