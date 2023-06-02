import CreateIcon from '@mui/icons-material/Create'
import PersonIcon from '@mui/icons-material/Person'
import { ThemeProvider } from '@mui/material'
import { motion } from 'framer-motion'
import React from 'react'

import { AvatarProfile, customFontIconHeader } from '../../themes/profileStyle'

type Props = {
  profilePicture?: string
  onClickBtn: () => void
}

const AvartarCustomize = (props: Props) => {
  const { profilePicture, onClickBtn } = props
  return (
    <div className='avatar__div--containers'>
      <div className='avatar__div--edit'>
        <ThemeProvider theme={customFontIconHeader}>
          <AvatarProfile src={profilePicture}>
            <PersonIcon sx={{ fontSize: '60px' }} />
          </AvatarProfile>
          <motion.button
            whileTap={{ scale: 1.2 }}
            className='avatar__btn--edit'
            onClick={onClickBtn}
          >
            <CreateIcon fontSize='small' />
          </motion.button>
        </ThemeProvider>
      </div>
    </div>
  )
}

export default AvartarCustomize
