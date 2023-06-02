import { CircularProgress, Stack } from '@mui/material'

import '../assets/css/components/loading.css'

const logo = require('../assets/img/logo-hospital-care.png')

const Loading = () => {
  return (
    <div className='loading__box'>
      <Stack
        className='loading__progress'
        sx={{ color: 'grey.500' }}
        spacing={2}
        direction='row'
      >
        <CircularProgress color='inherit' />
      </Stack>
      <div className='loading__box--logo'>
        <img className='loading__slogan--img' src={logo} alt='' />
        <p className='loading__slogan--text'>Hospital Care</p>
      </div>
    </div>
  )
}

export default Loading
